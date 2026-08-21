# GLM5-next MTP verify 图化设计 (Stage B)

状态: 设计定稿 (2026-08-21), 基于 Stage A 实验与 kernel 语义实测
目标: MTP verify forward 走 aclgraph → TPOT ~40-42ms (单并发), 满足 ≤50ms 指标

## 1. 背景与根因 (实测数据)

| 配置 | TPOT | 说明 |
|---|---|---|
| nomtp + graph | 62.5ms | decode 走图 |
| nomtp + eager | 148ms | 45 层 Python launch 开销 |
| MTP (现状) | 119ms | verify 全程 eager (600/0), 接受率 68%, 1.68 tok/步 |

劣化根因: verify 无法入图。阻塞点:
1. KDA verify 的状态管理是 host 逻辑 (npu_paged_attention.py 的 `_mtp_pending`
   Python 状态机 + per-seq for 循环 + 每步 torch.tensor H2D)
2. 图 replay 只能重放固定 kernel 序列 → 一切每步变化的量必须变成 device 张量值

## 2. 当前 lazy-advance 语义 (必须逐位保持)

每 KDA 层每 verify 步 (num_spec=1), 行布局 [b, d] (b=上步已确认 token 重处理, d=起草):

1. `pending[slot]` 暂存上一步的 raw qkv/gate/beta 行 `[b_prev, d_prev]` 与 prev_base
2. `m = base_now - prev_base ∈ {1,2}` (kv 增量; 上步接受→2, 拒绝→1)
3. advance: 取 stash 前 m 行, conv 推进 + recurrent_kda 推进 live state (output_final_state=True)
4. verify 行 [b,d] 只读链式 (output_final_state=False) 出 logits 行
5. 当前行入 stash 供下步

## 3. 图化设计 (全部固定 shape + device 张量)

### 3.1 关键技巧: m-选择 → mask

recurrent 链中 gate=0 且 beta=0 的行: decay=exp(0)=1, delta=0 → **状态零贡献**。
因此 advance 恒为 [2 行/seq] varlen 调用 (cu=[0,2,4,...] 固定),
row1 的 gate/beta 用 device mask `(m==2)` 置零 → 最终态 = 前 m 行后态, 语义精确。

### 3.2 每 KDA 层每步的图内算子序列

```
持久缓冲 (per backend, per bucket B):
  stash_raw  [B, conv_dim, 2]     上步 raw mixed_qkv 行 (index_copy 写入 idx)
  stash_g/b  [B, 2, H, D]         上步 gate/beta raw
  tails      [2, B, conv_dim, K-1]  conv 状态双尾 (after-b / after-bd)
  kv_pos_prev [B]                  上步 kv 位置 (device)
  cu2 = [0,2,4,...]                固定 cu (bucket)

每步 device 输入: cur rows [b,d], idx [B], kv_pos_cur [B]
  m = clamp(kv_pos_cur - kv_pos_prev, 1, 2)        # device

1) conv 链 (纯 torch, 固定 shape):
   cs = select(tails, m-1, idx)                    # gather: after-b / after-bd
   chained = conv1d(cat([cs, cur_qkv], -1))         # [B, dim, 2+K-1] → [B, dim, 2]
   tails_new[0] = window 尾 after-b; tails_new[1] = after-bd
   tails ← scatter(tails_new, idx)                   # 两个尾都写, 下步选
2) recurrent advance (被 mask 的双行链):
   adv_in = (stash_raw, stash_g, stash_b) 的 conv 后 qkv (同 1) 的 stash 版本
   g_masked = stash_g * (m==2 on row1); b_masked = stash_b * (m==2 on row1)
   st = ssm_pool.index_select(0, idx)
   _, st_adv = recurrent_kda(adv_q/k/v/g_masked/b_masked, initial=st,
                             cu=cu2, output_final_state=True, state_v_first=True)
   ssm_pool.index_copy_(0, idx, st_adv)
3) verify 只读链 (现 commit_first_only 分支保留, 去 seqwise 循环):
   out2 = recurrent_kda(cur rows conv 后, initial=ssm_pool.index_select(0, idx),
                        cu=cu2, output_final_state=False)
4) stash 更新: stash_raw ← cur rows (index_copy)
```

### 3.3 conv 的窗口正确性

conv 无 mask 技巧 (滑窗不可屏蔽), 用双尾 staging:
当前步的 conv 链起点 = select(tails, m_prev-1) — 上步双尾按上步 m 选。
tails 每步双写 [after-b, after-bd], 下步按新 m gather。

### 3.4 m 的来源

m = kv_seq_lens 增量, 可 device 计算 (kv_cur - kv_prev) 或由 C++ 拒绝采样直出
(mtp_worker 已有 pending_target_context_.accepted_tokens device 张量)。

## 4. kernel 依赖 (Step 0 实测结论)

fla_npu recurrent_kda:
- ✅ varlen cu_seqlens (device 读值, 兼容 ACLGraph replay)
- ✅ 1D per-token slots / index_select+index_copy 模式
- ❌ 2D speculative 多 token (accepted=1 输出错误, 中间态写槽垃圾) — 上游 bug,
  不依赖。conv 为纯 torch F.conv1d, 无需 fla_npu kernel。

## 5. 实施步骤

- B1: npu_paged_attention.py 新增 env 门控分支 (GLM5_KDA_VERIFY_V2=1),
  eager 下跑 V2 路径, A/B 验证输出与现路径一致 (quality_battery.sh + 手工 curl)
- B2: Python DecodeAclGraphRunner 接受 verify 批 (bucket padding, can_execute 放行);
  C++ mtp_worker 送固定尺寸批; full-attn 层用 Qwen3.5 expanded-decode 机制
  (Stage A 已打通 populate 修复)
- B3: replay 的 device 侧更新 (persistent buffer + update_decode_graph_metadata)
- B4: draft (18ms) 保持 eager, 与 target 图重叠

## 6. 风险与回归

- 精度: 新调用边界可能触发 NPU tiling 1-ULP 漂移 (MTP 分支历史问题),
  必须 A/B: V2-eager vs 现路径 逐 token 比对
- mask 行 (gate=0,beta=0) 的数值路径需验证与真 1 行调用 bit 一致性
  (若不一致, B1 阶段即暴露, 有 fallback: cu 单行变体走 bucket=2 图)
- 上游 fla_npu bug 修复后可切换单调用模式 (语义已验证)

## 7. B1 实施进展 (2026-08-21, 数值已收口)

### 已完成
- `_spec_verify_v2` 方法 + 双 dispatch (verify 批 / plain 步), env 门控
  `GLM5_KDA_VERIFY_V2=1`, 默认关闭零风险
- 调试修掉的坑 (复现都需真实引擎):
  1. mixed_qkv [1,C,T] 布局: view→reshape(C,B,R).permute(1,0,2)
  2. NPU cat 拒绝非连续/stride-0 广播: 全部 .contiguous() 物化
  3. torch.where 广播条件 stride-0 → gather + 物化索引
  4. **F.conv1d 第4位置参数是 stride** (padding 必须关键字传)
  5. **NPU depthwise conv 拒绝 padding=0 形态** ("non-positive stride"),
     必须用 padding=Ks + 切 result[Ks:Ks+R] 的形态
  6. 嵌入式解释器 stderr 被吞, 调试输出必须写文件 (/tmp/v2dbg.log)
- mask 技巧离线 bit-exact 验证 ✓ (gate=0+beta=0 行 = 状态零操作)

### 数值对齐结论 (2026-08-21, per-layer abs-sum 双侧打点)
方法: 旧路径 GLM5NEXT_DEBUG_LINEAR=1 ([linear-debug-in/out] 落 node log)
+ V2 加同锚点同 key 打点 (GLM5_KDA_VERIFY_V2_NUMDBG=1 → /tmp/v2numdbg.log),
同 eager + HCCL_DETERMINISTIC, canonical curl (max_tokens=100)。

发现:
1. **verify step 0: 全部 34 个 KDA 层锚点完全一致** — V2 首步
   (boundary 读 cache、conv、只读链、stash) bit 级正确。
2. step 1 起 raw 投影输入 (mqkv/gate/beta) 漂移 ~1e-4 相对量, 但 L0 的
   advance 输入完全相同 → 漂移源于 advance 后状态的 ULP 级不可见差异经
   mHC sinkhorn (~16x/层) 放大。
3. **旧路径自身同样形态敏感**: 关掉 coordinator (GLM5_KDA_NO_COORD=1,
   逐层 advance, 调用形状与 V2 完全一致) 后, nocoord-old 与 V2 在
   step 0-1 全部锚点一致、step 2 起同样 ULP 漂移; 而 nocoord vs coord
   两个旧配置之间也互相分歧 — 与 _KDA_SEQWISE 注释记载的
   "flattened 多 segment 调用 in-process 选不同 tiling → 1-ULP 漂移"
   同一类问题。
4. 结论: **V2 无语义 bug**; 三个配置 (coord/nocoord/v2) 前 ~40 token
   (char 93) 一致后各走各的连贯分支, 属 ULP 噪声等价类。之前
   "token 2 即分歧" 的观察是混杂变量: 当时基线是 graph=True (plain 步
   走图内 manual conv), V2 是 eager (F.conv1d), 数值不同源。
5. B1 锚点注意: 旧路径 in 打点在 coordinator 之后/逐层 advance 之前,
   nocoord 模式下 in 锚点与 V2 错位一层 advance (OUT 锚点才可比)。

### B1 遗留 (并入 B2 的 persistent buffer 设计)
- V2 stash 按 batch 位置索引而非 slot — B>1 或批序变化会读错行;
  需改为 slot 键控 persistent buffer (index_copy_ by idx)
- V2 未处理 has_initial_state 冷启动清零 (旧路径 torch.where warm)
- prefill 时 `_v2_states.pop(layer_id)` 整层重置 — 并发场景会误伤其它
  seq 的 stash, 应改为按 prefill 批的 slot 置 armed=False

### 测试资产
- /tmp/kda_ab_logs/: old/v2/nocoord 三配置 node log + numdbg + 输出
- /tmp/kda_ab_old.sh / kda_ab_v2.sh / kda_ab_nocoord.sh: A/B 启动脚本
  (eager + HCCL_DETERMINISTIC + DEBUG_LINEAR)
- /tmp/kda_diff2.py: per-layer 锚点 diff 脚本
- /tmp/mtp_ab/: 早期 4-prompt 基线 (注意: 基线为 graph=True, 有混杂)
- /tmp/test_kda_final.py / test_mask_trick.py: kernel 语义/mask 验证脚本
