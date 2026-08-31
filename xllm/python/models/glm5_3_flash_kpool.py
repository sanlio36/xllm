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
    """与 Glm53FlashIndexer.get_pooled_states 全量输出逐 bit 相同。"""
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


# ---------------------------------------------------------------------------
# Paged pool cache: write-time incremental compression + direct read.
#
# Physical blocks reuse the token block table: pool logical block L covers
# pools [L*bs/rate, (L+1)*bs/rate) = tokens [L*bs, (L+1)*bs) = token logical
# block L, so pool_slot(p) = bt[s, p//(bs/rate)] * (bs/rate) + p%(bs/rate).
# addressing without engine-side pool-granular allocation. The 257-wide
# token-granular index cache is kept untouched: compression inputs are read
# from it, which also solves chunked-prefill pools spanning chunk boundaries
# (vllm's dual-cache design).
# ---------------------------------------------------------------------------

def alloc_pool_cache(index_cache: torch.Tensor, rate: int) -> torch.Tensor:
    """Allocate a pool-granularity cache matching the configured pool rate."""
    if rate < 1:
        raise ValueError("pool rate must be positive")
    bs = index_cache.shape[1]
    if bs % rate != 0:
        raise ValueError(
            f"block_size {bs} must be divisible by index_kpool {rate}"
        )
    if index_cache.shape[-1] < 3 or (index_cache.shape[-1] - 1) % 2 != 0:
        raise ValueError("compressed index cache width must be 2 * head_dim + 1")
    head_dim = (index_cache.shape[-1] - 1) // 2
    return torch.zeros(
        index_cache.shape[0], bs // rate, 1, head_dim,
        dtype=index_cache.dtype, device=index_cache.device,
    )


def compress_completed_pools(index_cache: torch.Tensor,
                             pool_cache: torch.Tensor,
                             block_table: torch.Tensor,
                             positions: torch.Tensor,
                             ape: torch.Tensor, head_dim: int, rate: int) -> None:
    """把本步完成 pool 的压缩 k 写入池 cache。

    ``block_table`` 是**单序列**一行 ``[1, n_logical_blocks]``；``positions``
    是该序列本步写入 token 的绝对位置。数学与 ``pooled_states()`` 完全
    同序（fp32 softmax + bf16 逐积 round），写入值 == 旧路径每步重算值
    （逐 bit）。graph 安全：无数据依赖形状，未完成 pool 经 write 掩码
    保留原值。
    """
    bs = index_cache.shape[1]
    pool_bs = pool_cache.shape[1]
    width = 2 * head_dim + 1
    flat = index_cache.reshape(-1, width)
    pool_flat = pool_cache.reshape(-1, head_dim)
    n_tok = positions.shape[0]
    rate_off = torch.arange(rate, device=positions.device)
    # 每个 token 是其所属 pool 的末 token: pool p = pos // rate
    pos = positions.reshape(-1, 1)                       # [N, 1]
    member_pos = pos - (rate - 1) + rate_off[None, :]    # [N, rate]
    done = ((pos + 1) % rate == 0) & (pos >= rate - 1)   # [N, 1]
    member_valid = done & (member_pos >= 0)
    # member token 行: bt[t // bs] * bs + t % bs
    bt = block_table.reshape(-1)                         # [nblk]
    blk_idx = (member_pos.clamp(min=0) // bs).clamp(max=bt.shape[0] - 1)
    slots = bt[blk_idx] * bs + member_pos.clamp(min=0) % bs   # [N, rate]
    rows = flat[slots.reshape(-1)].reshape(n_tok, rate, width)
    gate = rows[..., head_dim:2 * head_dim].float()
    logits = gate + ape.float()[None]                    # ape [rate, head_dim]
    logits = logits.masked_fill(~member_valid[..., None], float("-inf"))
    cache_dtype = pool_cache.dtype
    weights = torch.nan_to_num(logits.softmax(1)).to(cache_dtype)
    keys = rows[..., :head_dim].float()
    # Mirror pooled_states' per-product cache-dtype rounding and fp32 sum.
    prod = (weights.float() * keys).to(cache_dtype).float()
    compressed = prod.sum(1).to(cache_dtype)             # [N, head_dim]
    # 写入槽位: bt[p // pool_bs] * pool_bs + p % pool_bs
    pool_id = (pos // rate).reshape(-1)                  # [N]
    pool_blk = (pool_id // pool_bs).clamp(max=bt.shape[0] - 1)
    pool_slots = bt[pool_blk] * pool_bs + pool_id % pool_bs
    pool_slots = pool_slots.clamp(0, pool_flat.shape[0] - 1)
    write = done.reshape(-1)
    if n_tok > 1:
        # Prefill chunk (eager): 同一 pool 的多个 token 都产生写请求,
        # 重复索引的原位赋值非确定(非 done 的旧值写回会覆盖 done 的压缩值),
        # 先过滤到仅完成 pool。decode graph 走 n_tok==1 分支(无重复, 静态形状)。
        sel = write.nonzero().flatten()
        pool_flat.index_copy_(0, pool_slots[sel], compressed[sel])
    else:
        # masked 原位写(graph 静态形状): 未完成 pool 保留原值
        pool_flat[pool_slots] = torch.where(
            write[:, None], compressed, pool_flat[pool_slots]
        )


def read_pools(pool_cache: torch.Tensor, block_table: torch.Tensor,
               kv_lens: torch.Tensor, n_pools: int, rate: int):
    """直读池 cache -> ``(pool_keys [B,P,D], pool_indices [B,P,rate] int64,
    pool_valid [B,P] bool)``。

    ``pool_valid = 4(p+1) <= kv_len`` 与旧路径 "4 slot 全 valid" 等价；
    ``pool_indices`` 的 -1 掩码沿用旧 ``slot_valid`` 语义
    (``4p+s >= kv_len[b]`` 置 -1, 按每序列)。
    """
    pool_bs = pool_cache.shape[1]
    head_dim = pool_cache.shape[-1]
    device = pool_cache.device
    B = block_table.shape[0]
    pool_flat = pool_cache.reshape(-1, head_dim)
    offs = torch.arange(n_pools, device=device)
    # [B, P] = bt[b, p // pool_bs] * pool_bs + p % pool_bs
    blk = (offs // pool_bs).clamp(max=block_table.shape[1] - 1)
    pool_slots = block_table[:, blk].clamp(min=0) * pool_bs + offs[None, :] % pool_bs
    pool_keys = pool_flat[pool_slots.reshape(-1)].reshape(B, n_pools, head_dim)
    slot_off = torch.arange(rate, device=device)
    member = offs[:, None] * rate + slot_off[None, :]    # [P, rate]
    # -1 掩码与旧 slot_valid(key_valid & in_range) 对齐: 按每序列 kv_len 掩
    member_b = member[None].expand(B, -1, -1)
    pool_indices = member_b.masked_fill(
        member_b >= kv_lens.reshape(-1, 1, 1), -1).contiguous()
    pool_valid = (offs[None, :] + 1) * rate <= kv_lens.reshape(-1, 1)
    return pool_keys, pool_indices, pool_valid
