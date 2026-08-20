# GLM5-next mHC 融合算子接入设计文档

## 1. 背景

### 1.1 什么是 mHC

GLM5-next 模型使用 **Manifold-constrained Hyper-Connection (mHC)** 机制，在每层 decoder layer 中维护 4 路残差流（`hc_mult=4`），通过可学习的 collapse/expand 权重、sigmoid gate 和 Sinkhorn 归一化来控制多路残差之间的信息流动。

每个 decoder layer 包含两个 mHC 站点：
- **Attn mHC**：在 self-attention 前后，collapse 4 路残差为 1 路 → attention → expand 回 4 路
- **FFN mHC**：在 FFN/MLP 前后，同样 collapse → FFN → expand

### 1.2 纯 Python 实现的性能问题

原始实现中，`Glm5NextHyperConnection.forward()` 包含以下纯 Python 操作：

1. **Unweighted RMSNorm**：计算 `rsqrt(mean(x^2))`
2. **Linear 投影**：大矩阵乘法 `[B, S, hc*D] @ [mix, hc*D]^T`
3. **Split + sigmoid gates**：`torch.sigmoid(pre_w * scale + bias)`
4. **Sinkhorn 归一化**：20 次迭代的 row/column normalization（每次都在 fp32 下进行）
5. **Weighted sum**：`(pre * x).sum(dim=hc)` 做 collapse

这些操作每次 forward 需要多次 kernel launch，Sinkhorn 迭代在 Python 层做循环，性能开销大。

## 2. vLLM 参考实现

vLLM 已在 `vllm-ascend-glm-next` 中接入了 mHC 融合算子，核心代码位于 `vllm_ascend/models/glm5_next.py`。

### 2.1 vLLM 的 hc_pre

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    n = self.hc_mult
    d = self.hidden_size
    x_mhc = x.view(x.shape[0], n, d)          # [tokens, hc_mult, D]
    y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x_mhc, hc_fn, hc_scale, hc_base,
        n, self.hc_sinkhorn_iters, self.rms_norm_eps, self.hc_eps,
    )
    return y, x_mhc, post, comb
```

- 输入 `x`：`[tokens, hc_mult * D]`，reshape 为 `[tokens, hc_mult, D]`
- 单次调用完成：rsqrt + linear + sigmoid + Sinkhorn + weighted-sum-reduce
- 返回 `y [tokens, D]`, `residual [tokens, hc_mult, D]`, `post [tokens, hc_mult]`, `comb [tokens, hc_mult, hc_mult]`

### 2.2 vLLM 的 hc_post

```python
def hc_post(self, x, residual_mhc, post, comb):
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0),           # [1, tokens, D]
        residual_mhc.unsqueeze(0), # [1, tokens, hc_mult, D]
        post.unsqueeze(0),        # [1, tokens, hc_mult]
        comb.unsqueeze(0),        # [1, tokens, hc_mult, hc_mult]
    ).squeeze(0)
```

- 单次调用完成：`post * x + comb^T @ residual`
- vLLM 无 sequence 维度（per-token 处理），通过 `unsqueeze(0)` 补 batch 维

### 2.3 vLLM 的参数存储

```python
self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
```

关键点：`hc_scale` 和 `hc_base` 必须以 **float32** 存储，因为底层 ACLNN 算子要求 float32 输入。

## 3. xLLM 接入方案

### 3.1 总体思路

xLLM 底层已有 NPU 融合 kernel（`hc_pre` / `hc_post`），实现在 `xllm/core/kernels/npu/xllm_ops/` 下。接入方案：

1. 通过 pybind11 将 C++ kernel 暴露到 `xllm_runtime` Python 模块
2. 修改 `glm5_next.py` 中的 `Glm5NextHyperConnection` 和 `Glm5NextDecoderLayer`，用融合算子替代纯 Python 操作
3. 通过 feature flag 实现融合路径/原始路径的 fallback 切换

### 3.2 新增文件

#### `xllm/core/runtime/py_hyper_connection.h`

声明 `register_hyper_connection_kernels` 函数。

#### `xllm/core/runtime/py_hyper_connection.cpp`

pybind11 绑定，将 NPU kernel 暴露为 `xllm_runtime` 模块的方法：

```cpp
void register_hyper_connection_kernels(py::module_& m) {
  // hc_pre: fused rsqrt + linear + sinkhorn + weighted-sum-reduce
  m.def("hc_pre",
        &kernel::npu::hc_pre,
        py::arg("x"),          // [B, S, hc_mult, D]
        py::arg("hc_fn"),      // linear weight [mix, hc_mult*D]
        py::arg("hc_scale"),   // scale [3], float32
        py::arg("hc_base"),    // bias [mix], float32
        py::arg("hc_mult"),    // number of residual streams
        py::arg("hc_sinkhorn_iters"),
        py::arg("norm_eps"),
        py::arg("hc_eps"));

  // hc_post: post * x + comb^T @ residual
  m.def("hc_post",
        &kernel::npu::hc_post,
        py::arg("x"),          // [B, S, D]
        py::arg("residual"),   // [B, S, hc_mult, D]
        py::arg("post"),       // [B, S, hc_mult]
        py::arg("comb"));      // [B, S, hc_mult, hc_mult]
}
```

### 3.3 修改文件

#### `xllm/core/runtime/py_executor_impl.cpp`

在 `PYBIND11_EMBEDDED_MODULE(xllm_runtime, m)` 中注册 hc kernel：

```cpp
#if defined(USE_NPU)
#include "core/runtime/py_hyper_connection.h"
#endif

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) {
  register_attention_metadata_views(m);
#if defined(USE_NPU)
  register_hyper_connection_kernels(m);
  // ...
#endif
}
```

#### `xllm/core/runtime/CMakeLists.txt`

添加 `py_hyper_connection.h` 和 `py_hyper_connection.cpp`，通过 `USE_NPU` 条件编译。

#### `xllm/python/models/glm5_next.py`

##### 3.3.1 Feature Flag

模块级 feature flag，运行时检测融合算子是否可用：

```python
try:
    import xllm_runtime
    _has_mhc_fused = hasattr(xllm_runtime, "hc_pre")
except ImportError:
    xllm_runtime = None
    _has_mhc_fused = False
```

- `xllm_runtime` 不可用 → 回退到纯 Python 参考实现
- `xllm_runtime` 可用但没有 `hc_pre` → 也回退
- 两边都可用 → 走融合路径

##### 3.3.2 参数 dtype 对齐

与 vLLM 对齐，`hc_scale` 和 `hc_base` 在 `__init__` 中声明为 float32：

```python
self.fn = nn.Parameter(
    torch.empty(mix, self.hc_mult * cfg.hidden_size, dtype=torch.float32, device=device))
self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32, device=device))
self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))
```

由于 weight loader 在加载时可能将参数转回 bf16，在调用 kernel 时额外加 `.float()` 守卫：

```python
collapsed, post, comb = xllm_runtime.hc_pre(
    hidden_streams,
    self.fn,
    self.scale.float(),   # 防止 weight loader 转回 bf16
    self.base.float(),
    ...
)
```

##### 3.3.3 Glm5NextHyperConnection.forward() 双路径

```python
def forward(self, hidden_streams):
    if _has_mhc_fused:
        # 融合路径：单次 kernel 调用
        collapsed, post, comb = xllm_runtime.hc_pre(
            hidden_streams, self.fn,
            self.scale.float(), self.base.float(),
            hc, self.hc_sinkhorn_iters,
            self.input_norm.variance_epsilon, self.hc_eps,
        )
        return post, comb, collapsed
    else:
        # 参考路径：纯 Python 逐步计算
        ...
```

##### 3.3.4 Glm5NextDecoderLayer 双路径

```python
def forward(self, hidden_states, ...):
    if _has_mhc_fused:
        return self._forward_fused(hidden_states, ...)
    else:
        return self._forward_ref(hidden_states, ...)
```

- **`_forward_fused`**：用 `xllm_runtime.hc_post` 做 recombination
- **`_forward_ref`**：用 `post * x + comb^T @ residual` 做 recombination

### 3.4 xLLM vs vLLM 的关键差异

| 维度 | vLLM | xLLM |
|------|------|------|
| **Sequence 维度** | 无 S 维（per-token），`[tokens, D]` | 有 S 维，`[B, S, D]` |
| **hc_pre 输入** | `[tokens, hc_mult, D]`，手动 reshape | `[B, S, hc_mult, D]`，kernel 内部处理 |
| **hc_post 输入** | `[tokens, D]`，unsqueeze(0) 补 batch | `[B, S, D]`，直接传入 |
| **算子调用** | `torch.ops._C_ascend.npu_hc_pre_v2` | `xllm_runtime.hc_pre` |
| **底层实现** | 同一套 ACLNN 算子（`aclnnHcPreSinkhorn` 等） | 同一套 ACLNN 算子 |

### 3.5 MLA Attention 输出 reshape

xLLM 的 `Glm5NextMlaAttention.forward()` 将输入从 `[B, S, D]` 压平为 `[B*S, D]` 进行计算，返回 2D 输出。这导致 `hc_post` 收到的 `x` 为 2D，而 kernel 要求 3D。

解决方式：在 `_forward_fused` 和 `_forward_ref` 中，attention 输出后检查 dim，若为 2D 则 reshape 回 `[B, S, D]`：

```python
# MLA attention returns [B*S, D] (2D); reshape to [B, S, D].
if hidden_states.dim() == 2:
    hidden_states = hidden_states.view(
        residual.shape[0], residual.shape[1], -1)
```

> 注：`Glm5NextKdaAttention` 返回 3D `[B, S, D]`，不受此逻辑影响。

## 4. 数据流总览

### 4.1 融合路径 (hc_pre + hc_post)

```
Input: [B, S, hc_mult, D]
    │
    ▼
┌──────────────────────────────────────┐
│  attn_hc (hc_pre)                     │
│  [B,S,hc_mult,D] → [B,S,D] collapsed │
│  + post [B,S,hc_mult] + comb [B,S,hc_mult,hc_mult]
└──────────────────────────────────────┘
    │
    ▼
  input_layernorm → [B, S, D]
    │
    ▼
  self_attn (MLA/KDA) → [B, S, D] or [B*S, D]
    │
    │ (reshape if 2D)
    ▼
┌──────────────────────────────────────┐
│  hc_post                             │
│  x [B,S,D] + residual [B,S,hc_mult,D]│
│  + post [B,S,hc_mult] + comb [...]   │
│  → [B, S, hc_mult, D]                │
└──────────────────────────────────────┘
    │
    ▼
  ffn_hc (hc_pre) → [B, S, D] collapsed
    │
    ▼
  post_attention_layernorm → [B, S, D]
    │
    ▼
  mlp (Dense/MoE) → [B, S, D]
    │
    ▼
┌──────────────────────────────────────┐
│  hc_post                             │
│  → [B, S, hc_mult, D]                │
└──────────────────────────────────────┘
    │
    ▼
Output: [B, S, hc_mult, D]
```

### 4.2 参考路径 (纯 Python)

流程相同，仅 hc_pre 和 hc_post 替换为纯 Python 逐步计算。

## 5. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `xllm/core/runtime/py_hyper_connection.h` | 新增 | 声明 `register_hyper_connection_kernels` |
| `xllm/core/runtime/py_hyper_connection.cpp` | 新增 | pybind11 绑定，暴露 `hc_pre` / `hc_post` |
| `xllm/core/runtime/py_executor_impl.cpp` | 修改 | 添加 `#include` 和 `register_hyper_connection_kernels(m)` |
| `xllm/core/runtime/CMakeLists.txt` | 修改 | 添加 `py_hyper_connection.h/.cpp`（`USE_NPU` 条件编译） |
| `xllm/python/models/glm5_next.py` | 修改 | 添加 feature flag、双路径实现、参数 dtype 对齐、MLA reshape |

## 6. 测试验证

1. **单元测试**：运行 `deepseek_v4_hyper_connection_test` 验证底层 kernel 正确性
2. **精度验证**：对比融合路径与参考路径的输出 logits，误差应在 1e-2 (bf16) 以内
3. **性能验证**：对比融合路径与参考路径的 per-step latency，预期减少 kernel launch 次数和 Python 开销
4. **端到端测试**：启动服务，发送推理请求，确认输出正确且无崩溃