"""kPool 池化的快速 torch 实现（与原 get_pooled_states 逐 bit 等价）。

用 ``F.embedding``（[B, P, rate] 小索引）替换 flat 化的
``gather(1, flat_idx)``（[B, P*rate*D] int64 大索引）：被选取的元素集合
完全一致（safe 已 clamp），仅省掉 4.19M 元素索引张量的构建与随机访存，
微基准 2.50ms -> 0.05ms（B=1, T=32768, D=128, rate=4, npu:1）。
本模块保持纯 torch、无 xllm 依赖，便于单测直接加载。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gather_rows(src: torch.Tensor, safe_indices: torch.Tensor) -> torch.Tensor:
    """src [B, T, D]（可为 strided view），safe_indices [B, P, R] -> [B, P, R, D]。"""
    if src.shape[0] == 1:
        return F.embedding(safe_indices[0], src[0]).unsqueeze(0)
    return torch.stack(
        [F.embedding(safe_indices[b], src[b]) for b in range(src.shape[0])]
    )


def pooled_states(packed_states: torch.Tensor, key_valid: torch.Tensor,
                  ape: torch.Tensor, head_dim: int, rate: int):
    """与 Glm5NextIndexer.get_pooled_states 全量输出逐 bit 相同。"""
    keys, gate_scores, _ = torch.split(
        packed_states, [head_dim, head_dim, 1], dim=-1
    )
    batch_size, total_len = keys.shape[:2]
    device = keys.device
    first_key = torch.where(
        key_valid.any(-1),
        key_valid.to(torch.int32).argmax(-1),
        torch.full((batch_size,), total_len, dtype=torch.long, device=device),
    )
    n_pools = (total_len + rate - 1) // rate
    pool_offsets = torch.arange(n_pools, device=device) * rate
    slot_offsets = torch.arange(rate, device=device)
    pool_indices = (
        first_key[:, None, None]
        + pool_offsets[None, :, None]
        + slot_offsets[None, None, :]
    )
    slot_in_range = pool_indices < total_len
    safe_indices = pool_indices.clamp(0, total_len - 1)
    grouped_keys = _gather_rows(keys, safe_indices)          # [B, P, R, D]
    grouped_gate_scores = _gather_rows(gate_scores, safe_indices)
    slot_valid = key_valid.to(torch.uint8).gather(
        1, safe_indices.reshape(batch_size, -1)
    ).reshape(batch_size, n_pools, rate).to(torch.bool) & slot_in_range
    pool_valid = slot_valid.all(-1)
    logits = grouped_gate_scores.float() + ape.float()[None, None]
    logits = logits.masked_fill(~slot_valid[..., None], float("-inf"))
    weights = torch.nan_to_num(logits.softmax(2)).to(grouped_keys.dtype)
    pool_keys = (weights * grouped_keys).sum(2)
    pool_indices = pool_indices.masked_fill(~slot_valid, -1)
    return pool_keys, pool_indices, pool_valid
