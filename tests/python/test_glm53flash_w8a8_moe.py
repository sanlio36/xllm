"""CPU unit tests for Glm53FlashMoE W8A8 expert path (kernels stubbed)."""
import pytest
import torch

from tests.python.conftest import _install_python_package_stub  # noqa: F401
from xllm.python.models.glm5_3_flash import Glm53FlashConfig, Glm53FlashMoE


def _cfg(tp=1, n_experts=4, moe_inter=64, hidden=32, topk=2):
    return Glm53FlashConfig(
        hidden_size=hidden, moe_intermediate_size=moe_inter,
        n_routed_experts=n_experts, num_experts_per_tok=topk,
        n_group=1, topk_group=1, routed_scaling_factor=2.5,
        norm_topk_prob=True, tp_size=tp, n_layers=4,
        first_k_dense_replace=3,
    )


def test_moe_has_w8a8_int8_expert_params():
    cfg = _cfg(tp=1)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    inter_local = cfg.moe_intermediate_size // cfg.tp_size
    # int8 expert weights present
    assert moe.experts_w13.shape == (cfg.n_routed_experts, 2*inter_local, cfg.hidden_size)
    assert moe.experts_w13.dtype == torch.int8
    assert moe.experts_w2.shape == (cfg.n_routed_experts, cfg.hidden_size, inter_local)
    assert moe.experts_w2.dtype == torch.int8
    # scale/offset buffers
    assert moe.experts_w13_scale.shape == (cfg.n_routed_experts, 2*inter_local, 1)
    assert moe.experts_w13_scale.dtype == torch.float32
    assert moe.experts_w2_scale.shape == (cfg.n_routed_experts, cfg.hidden_size, 1)
    assert moe.experts_w2_offset.dtype == torch.float32
    # default flag
    assert moe.use_w8a8 is False
    # router + shared still present
    assert hasattr(moe, "gate") and hasattr(moe, "shared_experts")
    # bf16 experts still present (dual branch)
    assert hasattr(moe, "experts")


def test_moe_tp_shards_expert_intermediate():
    cfg = _cfg(tp=2, moe_inter=64)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    inter_local = cfg.moe_intermediate_size // cfg.tp_size  # 32
    assert moe.experts_w13.shape == (cfg.n_routed_experts, 2*inter_local, cfg.hidden_size)
    assert moe.experts_w2.shape == (cfg.n_routed_experts, cfg.hidden_size, inter_local)


def test_process_weights_w8a8_transposes_and_flattens(monkeypatch):
    cfg = _cfg(tp=1)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    moe.use_w8a8 = True
    # prepare_grouped_moe_weights 走 stub — 用 identity 让我们能断言 transpose 发生
    import xllm.python.kernels as K
    monkeypatch.setattr(K, "prepare_grouped_moe_weights",
                        lambda w13, w2: (w13, w2), raising=False)
    # 填非零 offset 会触发断言失败（对称约束）；这里填 0
    assert torch.all(moe.experts_w13_offset == 0)
    before13 = moe.experts_w13.shape
    before2 = moe.experts_w2.shape
    moe.process_weights_after_loading()
    # transpose(1,2): [n,2*inter,hidden] -> [n,hidden,2*inter]
    assert moe.experts_w13.shape == (before13[0], before13[2], before13[1])
    assert moe.experts_w2.shape == (before2[0], before2[2], before2[1])
    # scale flattened to [n, -1]
    assert moe.experts_w13_scale.dim() == 2
    assert moe.experts_w13_scale.shape[0] == cfg.n_routed_experts
    assert moe.experts_w2_scale.dim() == 2


def test_process_weights_bf16_branch_calls_shared_only(monkeypatch):
    cfg = _cfg(tp=1)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    moe.use_w8a8 = False
    called = {"shared": False}
    def _shared():
        called["shared"] = True
    monkeypatch.setattr(moe.shared_experts, "process_weights_after_loading", _shared, raising=False)
    # bf16: must NOT touch int8 expert weights (no transpose), must call shared
    before = moe.experts_w13.shape
    moe.process_weights_after_loading()
    assert moe.experts_w13.shape == before  # unchanged
    assert called["shared"] is True

# --- Task 3: forward dual branch ---

def test_forward_w8a8_branch_calls_grouped_moe(monkeypatch):
    cfg = _cfg(tp=1)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    moe.use_w8a8 = True
    calls = {}
    def _fake_grouped_moe(hidden, logits, w13, w2, s13, s2, bias, topk, tg, ng, renorm):
        calls["called"] = True
        calls["topk"] = topk
        calls["ng"] = ng
        calls["renorm"] = renorm
        calls["hidden_shape"] = tuple(hidden.shape)
        return torch.zeros(hidden.shape[0], cfg.hidden_size, dtype=torch.bfloat16)
    import xllm.python.kernels as K
    monkeypatch.setattr(K, "grouped_moe", _fake_grouped_moe, raising=False)
    monkeypatch.setattr(moe.shared_experts, "forward",
                        lambda x: torch.zeros_like(x), raising=False)
    h = torch.randn(2, 3, cfg.hidden_size, dtype=torch.bfloat16)
    out = moe(h)
    assert calls.get("called") is True
    assert calls["topk"] == cfg.num_experts_per_tok
    assert calls["ng"] == cfg.n_group
    assert calls["renorm"] is True
    assert calls["hidden_shape"] == (6, cfg.hidden_size)  # [2*3, hidden]
    assert out.shape == h.shape


def test_forward_bf16_branch_uses_old_experts(monkeypatch):
    cfg = _cfg(tp=1)
    moe = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
    # bf16 experts are lazily constructed — build them now (bf16 path only).
    from xllm.python.models.glm5_3_flash import Glm53FlashExperts
    moe.experts = Glm53FlashExperts(cfg, torch.bfloat16, torch.device("cpu"))
    moe.use_w8a8 = False
    grouped_called = {"v": False}
    import xllm.python.kernels as K
    monkeypatch.setattr(K, "grouped_moe",
                        lambda *a, **k: grouped_called.update(v=True) or torch.zeros(1, 1),
                        raising=False)
    experts_called = {"v": False}
    orig_forward = moe.experts.forward
    def _spy(h, ti, tw):
        experts_called["v"] = True
        return orig_forward(h, ti, tw)
    monkeypatch.setattr(moe.experts, "forward", _spy)
    monkeypatch.setattr(moe.shared_experts, "forward",
                        lambda x: torch.zeros_like(x), raising=False)
    h = torch.randn(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    moe(h)
    assert experts_called["v"] is True
    assert grouped_called["v"] is False  # bf16 must not call grouped_moe


# --- Task 4/5: load branch ---

import torch.nn as nn
from xllm.python.layers.qlinear import QLinearWeightLoader
from xllm.python.models.glm5_3_flash import Glm53FlashForCausalLM


class _FakeSD:
    """Minimal state-dict: name -> tensor."""
    def __init__(self, mapping):
        self._m = mapping
    def has(self, name):
        return name in self._m
    def get_tensor(self, name):
        return self._m[name]


def _w8a8_expert_tensors(mlp_pfx, n, moe_inter, hidden):
    """Build fake int8 expert tensors (weight+scale+offset) for n experts."""
    sd = {}
    for j in range(n):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if proj == "down_proj":
                shape = (hidden, moe_inter)
            else:
                shape = (moe_inter, hidden)
            sd[mlp_pfx + f"experts.{j}.{proj}.weight"] = torch.randint(
                -127, 127, shape, dtype=torch.int8)
            sd[mlp_pfx + f"experts.{j}.{proj}.weight_scale"] = torch.ones(
                shape[0], 1, dtype=torch.float32) * 0.01
            sd[mlp_pfx + f"experts.{j}.{proj}.weight_offset"] = torch.zeros(
                shape[0], 1, dtype=torch.float32)
    return sd


class _DummyModel(Glm53FlashForCausalLM):
    """Lightweight Glm53FlashForCausalLM shell: skip heavy construction, expose
    model.layers[i].mlp so _load_mlp (inherited) + helpers run on the MoE."""
    def __init__(self, cfg):
        nn.Module.__init__(self)  # bypass Glm53FlashForCausalLM.__init__
        self.cfg = cfg
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            layer = nn.Module()
            layer.mlp = Glm53FlashMoE(cfg, torch.bfloat16, torch.device("cpu"))
            self.model.layers.append(layer)


def test_load_mlp_w8a8_sets_flag_and_fills_int8(monkeypatch):
    cfg = _cfg(tp=1, n_experts=4, moe_inter=64, hidden=32)
    # process_weights_after_loading calls kernels.prepare_grouped_moe_weights —
    # stub it (conftest stubs kernels as empty module, so raising=False).
    import xllm.python.kernels as K
    monkeypatch.setattr(K, "prepare_grouped_moe_weights",
                        lambda w13, w2: (w13, w2), raising=False)
    dummy = _DummyModel(cfg)
    # find the first MoE layer, key the fake sd at that index
    moe_i = next(i for i in range(cfg.n_layers) if cfg.is_moe(i))
    mlp = f"model.layers.{moe_i}.mlp."
    sd = {}
    sd.update(_w8a8_expert_tensors(mlp, 4, 64, 32))
    sd[mlp + "gate.weight"] = torch.zeros(4, 32, dtype=torch.bfloat16)
    sd[mlp + "gate.e_score_correction_bias"] = torch.zeros(4, dtype=torch.float32)
    # shared_experts bf16 (probe -> fp): shared_inter = moe_inter * n_shared = 64*1
    for p in ("gate_proj", "up_proj"):
        sd[mlp + f"shared_experts.{p}.weight"] = torch.zeros(64, 32, dtype=torch.bfloat16)
    sd[mlp + "shared_experts.down_proj.weight"] = torch.zeros(32, 64, dtype=torch.bfloat16)
    L = QLinearWeightLoader(dummy, [_FakeSD(sd)], tp_size=1, tp_rank=0)
    dummy._load_mlp(L, mlp, moe_i)
    assert dummy.model.layers[moe_i].mlp.use_w8a8 is True
    exp = dummy.model.layers[moe_i].mlp.experts_w13
    assert exp.dtype == torch.int8
    assert exp.abs().sum() > 0  # filled, not empty
    assert dummy.model.layers[moe_i].mlp.experts_w2.abs().sum() > 0


def test_load_mlp_bf16_keeps_flag_false():
    cfg = _cfg(tp=1, n_experts=4, moe_inter=64, hidden=32)
    dummy = _DummyModel(cfg)
    moe_i = next(i for i in range(cfg.n_layers) if cfg.is_moe(i))
    mlp = f"model.layers.{moe_i}.mlp."
    sd = _w8a8_expert_tensors(mlp, 4, 64, 32)
    # remove weight_scale -> probe returns False -> bf16 path
    for k in list(sd):
        if "weight_scale" in k or "weight_offset" in k:
            del sd[k]
    # bf16 experts need real bf16 weights (not int8)
    for j in range(4):
        sd[mlp + f"experts.{j}.gate_proj.weight"] = torch.zeros(64, 32, dtype=torch.bfloat16)
        sd[mlp + f"experts.{j}.up_proj.weight"] = torch.zeros(64, 32, dtype=torch.bfloat16)
        sd[mlp + f"experts.{j}.down_proj.weight"] = torch.zeros(32, 64, dtype=torch.bfloat16)
    sd[mlp + "gate.weight"] = torch.zeros(4, 32, dtype=torch.bfloat16)
    sd[mlp + "gate.e_score_correction_bias"] = torch.zeros(4, dtype=torch.float32)
    for p in ("gate_proj", "up_proj"):
        sd[mlp + f"shared_experts.{p}.weight"] = torch.zeros(64, 32, dtype=torch.bfloat16)
    sd[mlp + "shared_experts.down_proj.weight"] = torch.zeros(32, 64, dtype=torch.bfloat16)
    # Pre-construct the bf16 experts so the loader's named_parameters() cache
    # (built in QLinearWeightLoader.__init__) sees experts.gate_up_proj.
    from xllm.python.models.glm5_3_flash import Glm53FlashExperts
    dummy.model.layers[moe_i].mlp.experts = Glm53FlashExperts(
        cfg, torch.bfloat16, torch.device("cpu"))
    L = QLinearWeightLoader(dummy, [_FakeSD(sd)], tp_size=1, tp_rank=0)
    dummy._load_mlp(L, mlp, moe_i)
    assert dummy.model.layers[moe_i].mlp.use_w8a8 is False
    assert dummy.model.layers[moe_i].mlp.experts.gate_up_proj.abs().sum() == 0
