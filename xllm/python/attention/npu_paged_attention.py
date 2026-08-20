# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import torch_npu

from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
    MlaPreprocessContext,
)
from xllm.python.model_executor.cp_utils import cp_gather_kv
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_execution_buffer,
    get_forward_context,
    get_forward_context_or_none,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext

# Spec-verify state-commit scheme: "lazy" (default) commits exactly the
# confirmed tokens - each verify step stashes its raw qkv/gate rows and the
# next step advances the live state by the kv delta (the count the C++
# actually committed: draft on accept, corrected argmax on reject), so the
# linear state tracks the true token stream exactly (verified offline
# against a plain-decode oracle: per-step output divergence decays to zero,
# vs a persistent ~6e-3 error for the alternative). "full"
# (GLM5_MTP_COMMIT=full) commits both verify rows every step - the
# re-processed row0 write is near-idempotent under the delta rule, but a
# rejected draft leaves a phantom write, a persistent per-step error.

_MTP_FULL_COMMIT = os.environ.get("GLM5_MTP_COMMIT", "lazy") == "full"
# Seq-wise KDA recurrent dispatch for spec-verify batches: split the flattened
# multi-sequence call into one single-segment call per sequence so concurrent
# verify runs the exact op-call shapes of single-request verify (see the
# read-only branch below for the divergence rationale).
_KDA_SEQWISE = os.environ.get("GLM5_KDA_SEQWISE", "1") == "1"
# Spec-verify acceptance instrumentation (MTP_TRACE=1): layer-0 counts
# verify steps and non-prefill plain steps (each reject inserts one plain
# bootstrap step, so the plain share of decode steps is the reject rate).
_MTP_TRACE = os.environ.get("MTP_TRACE", "") == "1"

# Ascend FIA sparse_mode values (see CANN aclnnFusedInferAttentionScore docs).
# 0: no compressed mask; used for single-query decode where no causal mask is
#    needed.
# 3: rightDownCausal; the causal mask is right-aligned to the KV tail, for the
#    prefix-cache / chunked-prefill case where q_len < kv_len so the new queries
#    attend the full cached prefix plus their own tokens (mode 2, leftUpCausal,
#    only aligns when q_len == kv_len and would misalign on a cache hit).
_SPARSE_MODE_NONE = 0
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3


def _mla_graph_max_seqlen_k(
    block_table: torch.Tensor,
    page_size: int,
) -> int:
    """Return a replay-stable KV length bound for MLA graph metadata."""
    max_seqlen_k = int(block_table.shape[1]) * int(page_size)
    if max_seqlen_k <= 0:
        raise RuntimeError("MLA graph block-table capacity must be positive")
    return max_seqlen_k


def _in_acl_graph() -> bool:
    """Whether the current forward runs under ACL graph warmup/capture.

    The decode graph runner always passes an ``execution_state`` (warmup and
    capture) and an ``acl_graph`` capture context (capture only); the eager
    runner sets neither, so eager paths stay byte-identical.
    """
    ctx = get_forward_context_or_none()
    return ctx is not None and (
        ctx.acl_graph is not None or ctx.execution_state is not None
    )


class NpuPagedAttentionBackend(AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device
        # MLA has separate dense/sparse-SFA execution paths, so the ordinary
        # MHA graph buffers must not be initialized for an MLA instance. The
        # pr2199 constructor heuristic (head_dim > 192 and num_kv_heads == 1)
        # covers DS V3.2 / glm5_2 (head_dim == qk_nope + qk_rope == 576) but
        # misses glm5_next, whose NoPE DSA layers report head_dim == 128 and
        # the model-level n_kv_heads; bind_kv_caches therefore refines the
        # flag from the actual cache layout (a latent-only key slot or a
        # paged sparse-index cache implies MLA).
        self._is_mla = head_dim > 192 and num_kv_heads == 1
        self._uses_sparse_mla = False

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._block_table_i32: torch.Tensor | None = None
        self._actual_seq_lens: list[int] | None = None
        self._actual_seq_q: list[int] = []
        self._actual_seq_kv: list[int] = []
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        self._mla_quant_indexer_metadata: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._mla_max_seqlen_q = 0
        self._mla_max_seqlen_k = 0
        self._graph_index_history_max_kv: int | None = None
        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1)
            .to(torch.int8)
            .contiguous()
            .to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        # Hybrid models (glm5_next) interleave KDA layers (key is None) with
        # DSA layers; scan for the first real KV cache instead of assuming
        # layer 0 is full attention.
        for cache in self._kv_caches:
            if cache.key is not None:
                return cache.key.shape[0]
        return 0

    @property
    def page_size(self) -> int:
        for cache in self._kv_caches:
            if cache.key is not None:
                return cache.key.shape[1]
        return 1

    @property
    def graph_index_history_max_kv(self) -> int:
        """Static KV-length cap for the kPool graph gather.

        The graph branch of ``gather_index_history`` densifies each sequence
        to a fixed ``[num_seqs, max_kv, width]`` buffer; sizing it by the full
        block-table capacity (max_position_embeddings can be 1M) is not
        viable. Decode steps whose block table exceeds this cap fall back to
        the eager runner (see DecodeAclGraphRunner), which keeps the dynamic
        gather. Override with XLLM_GRAPH_INDEX_HISTORY_MAX_KV.
        """
        if self._graph_index_history_max_kv is None:
            self._graph_index_history_max_kv = int(
                os.environ.get("XLLM_GRAPH_INDEX_HISTORY_MAX_KV", "32768")
            )
        return self._graph_index_history_max_kv

    @property
    def is_mla(self) -> bool:
        return self._is_mla

    @property
    def requires_host_kv_lengths(self) -> bool:
        """Whether ACL Graph replay must update FIA's host KV-length list."""
        return self._is_mla and not self._uses_sparse_mla

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches
        has_sparse_index = any(cache.index is not None for cache in kv_caches)
        # glm5_next DSA layers are NoPE: the latent lives in the key slot and
        # the value/rope slot is a 0-dim tensor normalized to None, while the
        # kPool indexer adds a paged index cache. Either signal marks this
        # backend instance as MLA even though the constructor heuristic
        # (head_dim > 192 and num_kv_heads == 1) does not fire for it.
        has_latent_only_cache = any(
            cache.key is not None and cache.value is None and cache.conv is None
            for cache in kv_caches
        )
        if has_sparse_index or has_latent_only_cache:
            self._is_mla = True
        self._uses_sparse_mla = self._is_mla and has_sparse_index

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        if metadata.q_cu_seq_lens is not None:
            self._actual_seq_lens = metadata.q_cu_seq_lens[1:].cpu().tolist()
        else:
            self._actual_seq_lens = None

        if metadata.block_table is not None:
            self._block_table_i32 = metadata.block_table.to(torch.int32)

            real_batch = metadata.block_table.shape[0]

            kv_host = metadata.kv_seq_lens_host
            if kv_host is not None:
                kv_host = kv_host.cpu()
                if kv_host.numel() == real_batch + 1:
                    per_seq_kv = kv_host[1:] - kv_host[:-1]
                else:
                    per_seq_kv = kv_host
            else:
                per_seq_kv = torch.ones(real_batch, dtype=torch.int32)

            kv_list = per_seq_kv[:real_batch].tolist()

            self._actual_seq_q = list(range(1, real_batch + 1))
            self._actual_seq_kv = kv_list
        else:
            self._block_table_i32 = None

        if (
            graph_mode
            and self._block_table_i32 is not None
            and not self._is_mla
        ):
            graph_batch_size = self._block_table_i32.shape[0]
            if self._graph_workspace is None:
                block_size = self.page_size
                dummy_q = torch.empty(
                    graph_batch_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                dummy_kv = torch.empty(
                    self.num_kv_blocks, block_size,
                    self.num_kv_heads * self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                self._graph_workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=self._actual_seq_q,
                        actual_seq_lengths_kv=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=_SPARSE_MODE_NONE,
                        scale=self.scale,
                        softmax_lse_flag=False,
                    )
                )
            if graph_batch_size not in self._graph_outputs:
                self._graph_outputs[graph_batch_size] = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._graph_lses[graph_batch_size] = torch.empty(
                    0, dtype=self.dtype, device=self.device
                )
            self._current_graph_output = self._graph_outputs[graph_batch_size]
            self._current_graph_lse = self._graph_lses[graph_batch_size]

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        # Gated on kv_seq_lens (not _is_mla) so the eager path is unchanged
        # for every model; the graph_mode sub-branches only swap the tensors
        # into static execution buffers and derive replay-stable bounds.
        self._mla_quant_indexer_metadata.clear()
        if metadata.kv_seq_lens is not None:
            kv_seq_lens = metadata.kv_seq_lens
            mla_device = kv_seq_lens.device
            actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if metadata.q_cu_seq_lens is not None:
                actual_seq_q = metadata.q_cu_seq_lens[1:].to(
                    torch.int32
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                actual_seq_q = torch.arange(
                    1, batch + 1, dtype=torch.int32, device=mla_device
                )
            if graph_mode:
                # ACL graph replay reuses the captured kernel arguments, so
                # the seq-lens must live in static buffers that the runner
                # rewrites before each replay.
                graph_batch = int(actual_seq_kv.numel())
                self._mla_actual_seq_q = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_Q", graph_batch),
                    lambda: torch.empty_like(actual_seq_q),
                )
                self._mla_actual_seq_kv = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_KV", graph_batch),
                    lambda: torch.empty_like(actual_seq_kv),
                )
                self._mla_actual_seq_q.copy_(actual_seq_q)
                self._mla_actual_seq_kv.copy_(actual_seq_kv)
            else:
                self._mla_actual_seq_q = actual_seq_q
                self._mla_actual_seq_kv = actual_seq_kv
            if metadata.is_prefill or metadata.is_chunked_prefill:
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                if q_seq_lens is not None and q_seq_lens.numel() > 0:
                    self._mla_max_seqlen_q = int(q_seq_lens.max().item())
                else:
                    seq_starts = torch.cat(
                        [actual_seq_q.new_zeros(1), actual_seq_q[:-1]]
                    )
                    self._mla_max_seqlen_q = int(
                        (actual_seq_q - seq_starts).max().item()
                    )
            else:
                self._mla_max_seqlen_q = 1
            if graph_mode and self._block_table_i32 is not None:
                # QuantLightningIndexer metadata is captured into the ACL
                # graph. Python scalar arguments are fixed at capture time,
                # while the device KV lengths continue to grow on replay.
                # Use the static graph block-table capacity as a safe bound so
                # the captured tiling metadata remains valid for every replay.
                self._mla_max_seqlen_k = _mla_graph_max_seqlen_k(
                    self._block_table_i32,
                    self.page_size,
                )
            else:
                self._mla_max_seqlen_k = int(actual_seq_kv.max().item())
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None
            self._mla_max_seqlen_q = 0
            self._mla_max_seqlen_k = 0

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        k_cache, v_cache = layer_cache.key, layer_cache.value
        if k_cache is None or v_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer_id}")
        num_tokens = q.shape[0]

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        # Context-Parallel prefill: q/k/v are this rank's sequence shard while
        # the slot_mapping/metadata still describe the full global sequence
        # (C++ does not pre-shard the Python path). All-gather K/V to the full
        # sequence, persist this rank's KV shard, and attend over its causal
        # prefix.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            return self._prefill_cp(
                q_3d, k_3d, v_3d, metadata, cp_context, k_cache, v_cache
            )

        kernels.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if metadata.is_prefill or metadata.is_chunked_prefill:
            return self._prefill(
                q_3d, k_3d, v_3d, k_cache, v_cache, metadata, num_tokens
            )
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor | None,
        k_pe_3d: torch.Tensor | None,
        layer: "Attention",
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        if topk is None:
            raise NotImplementedError(
                "dense MLA (topk=None) is not yet supported on "
                "NpuPagedAttentionBackend"
            )
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        # MLA reuses the K/V slots for the latent (nope) and rope caches.
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer_id}")
        # NoPE (qk_rope_head_dim==0): skip rope cache write + pass None to SFA.
        # The rope/value slot may be empty (a 0-dim tensor) or absent (None) in
        # NoPE models — it is never read, so do not require it.
        rope_dim = getattr(layer, "qk_rope_head_dim", None)
        if rope_dim and rope_dim > 0:
            if rope_cache is None:
                raise RuntimeError(
                    f"MLA rope cache is missing for layer {layer_id} "
                    f"(qk_rope_head_dim={rope_dim})")
            if not cache_is_preprocessed:
                if k_latent_3d is None or k_pe_3d is None:
                    raise RuntimeError("MLA cache inputs are required")
                torch.ops.xllm_ops.reshape_paged_cache(
                    metadata.slot_mapping, k_latent_3d, k_pe_3d, nope_cache, rope_cache
                )
            return self._mla_sparse(
                q_latent, q_pe, nope_cache, rope_cache, topk,
                self._block_table_i32, layer_id,
            )
        # NoPE path: latent only, no rope.
        if not cache_is_preprocessed:
            if k_latent_3d is None:
                raise RuntimeError("MLA cache inputs are required")
            torch.ops.xllm_ops.reshape_paged_cache(
                metadata.slot_mapping, k_latent_3d, k_latent_3d, nope_cache, nope_cache
            )
        return self._mla_sparse(
            q_latent, None, nope_cache, None, topk,
            self._block_table_i32, layer_id,
        )

    def mla_preprocess_context(
        self,
        layer: "Attention",
    ) -> MlaPreprocessContext | None:
        metadata = self._metadata
        if metadata is None or metadata.is_prefill or metadata.is_chunked_prefill:
            return None
        layer_cache = self._kv_caches[layer.layer_id]
        kv_cache, rope_cache = layer_cache.key, layer_cache.value
        if kv_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")
        return MlaPreprocessContext(
            kv_cache=kv_cache,
            rope_cache=rope_cache,
            slot_mapping=metadata.slot_mapping,
        )

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        assert self._block_table_i32 is not None
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        layer_cache = self._kv_caches[layer.layer_id]
        index_cache = layer_cache.index
        if index_cache is None:
            raise RuntimeError(
                f"MLA index cache is missing for layer {layer.layer_id}"
            )
        index_cache_scale = layer_cache.index_scale
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=metadata.slot_mapping,
            block_table=self._block_table_i32,
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
            index_cache_scale=index_cache_scale,
            get_quant_indexer_metadata=lambda num_heads_q,
            head_dim,
            sparse_count,
            cmp_ratio: self._get_quant_indexer_metadata(
                num_heads_q,
                index_cache.size(2),
                head_dim,
                sparse_count,
                cmp_ratio,
            ),
            update_index_cache=lambda values, scales: self._update_mla_index_cache(
                index_cache,
                index_cache_scale,
                metadata.slot_mapping,
                values,
                scales,
            ),
        )

    def _get_quant_indexer_metadata(
        self,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        sparse_count: int,
        cmp_ratio: int,
    ) -> torch.Tensor:
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        cache_key = (num_heads_q, head_dim, sparse_count, cmp_ratio)
        metadata = self._mla_quant_indexer_metadata.get(cache_key)
        if metadata is None:
            metadata = kernels.quant_lightning_indexer_metadata(
                num_heads_q,
                num_heads_k,
                head_dim,
                self._mla_actual_seq_q,
                self._mla_actual_seq_kv,
                self._mla_max_seqlen_q,
                self._mla_max_seqlen_k,
                sparse_count,
                cmp_ratio,
            )
            self._mla_quant_indexer_metadata[cache_key] = metadata
        return metadata

    @staticmethod
    def _update_mla_index_cache(
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> None:
        cache_view = index_cache.view(-1, index_cache.size(-1))
        scatter_indices = slot_mapping.reshape(-1, 1).clamp_min(0)
        kernels.scatter_nd_update(
            cache_view,
            scatter_indices,
            values,
        )
        if index_cache_scale is not None and scales is not None:
            scale_view = index_cache_scale.view(-1, index_cache_scale.size(-1))
            kernels.scatter_nd_update(scale_view, scatter_indices, scales)

    def gather_index_history(
        self, layer: "Attention", batch_size: int,
    ) -> torch.Tensor:
        """Gather the kPool packed history into a dense ``[B, kv_len, W]`` tensor.

        The index cache is paged (``[blocks, block_size, 1, W]``); the kPool
        indexer's ``select_topk`` needs the per-sequence rows contiguous. For
        each sequence we walk its block table, concatenating ``block_size``
        rows per block (the last block yields only ``last_page_len``).
        ``kv_len`` is padded to the max across the batch; out-of-range rows
        are zeroed so ``valid`` channels read as false downstream.
        """
        metadata = self._metadata
        assert metadata is not None, "gather_index_history called before prepare()"
        index_cache = self._kv_caches[layer.layer_id].index
        assert index_cache is not None, (
            "gather_index_history requires a paged index cache")
        block_table = metadata.block_table
        if block_table is None:
            # No paged view (standalone): caller should not reach here.
            raise RuntimeError("gather_index_history needs a paged block_table")
        block_size = index_cache.shape[1]
        width = index_cache.shape[3]
        device = index_cache.device

        # batch_size is hidden_states.shape[0] which the engine flattens to 1
        # for multi-sequence batches; the real sequence count is the block
        # table's first dim. Use it to gather every sequence's history.
        num_seqs = block_table.shape[0] if block_table is not None else batch_size

        if _in_acl_graph():
            # Graph branch: fixed shapes only (no .item()/host sync). Gather
            # the block table in one vectorized index_select up to a static
            # max_kv (replay-stable; capped by graph_index_history_max_kv —
            # the runner falls back to eager beyond it). Rows past each
            # sequence's live length are zeroed by an explicit kv_seq_lens
            # mask: the valid channel alone cannot be trusted because a
            # recycled block still carries a previous owner's valid=1 rows,
            # and a padded block-table column points at block 0.
            kv_lens_dev = metadata.kv_seq_lens
            if kv_lens_dev is None:
                raise RuntimeError(
                    "gather_index_history graph mode needs device kv_seq_lens")
            max_kv = min(
                block_table.shape[1] * block_size,
                self.graph_index_history_max_kv,
            )
            num_blocks = (max_kv + block_size - 1) // block_size
            out = get_execution_buffer(
                ("KPOOL_INDEX_HISTORY", num_seqs, max_kv, width),
                lambda: torch.empty(
                    num_seqs, max_kv, width,
                    dtype=index_cache.dtype, device=device,
                ),
            )
            flat = index_cache.view(-1, width)
            bt = block_table[:num_seqs, :num_blocks].to(torch.int64)
            block_offsets = torch.arange(block_size, device=device)
            slot_ids = (
                bt[:, :, None] * block_size + block_offsets[None, None, :]
            ).reshape(num_seqs, max_kv)
            gathered = flat.index_select(0, slot_ids.reshape(-1)).view(
                num_seqs, max_kv, width
            )
            row_valid = (
                torch.arange(max_kv, device=device)[None, :]
                < kv_lens_dev[:num_seqs].to(torch.int64)[:, None]
            )
            torch.mul(gathered, row_valid[:, :, None].to(out.dtype), out=out)
            return out

        kv_seq_lens = metadata.kv_seq_lens
        if kv_seq_lens is not None:
            kv_lens = kv_seq_lens[:num_seqs].to(torch.int64)
        else:
            kv_host = metadata.kv_seq_lens_host
            if kv_host is not None:
                kl = kv_host.cpu()
                if kl.numel() == num_seqs + 1:
                    kv_lens = (kl[1:] - kl[:-1]).to(torch.int64)
                else:
                    kv_lens = kl[:num_seqs].to(torch.int64)
            else:
                raise RuntimeError("gather_index_history needs kv_seq_lens")

        max_kv = int(kv_lens.max().item()) if num_seqs > 0 else 0
        flat = index_cache.view(-1, width)
        bt = block_table[:num_seqs].to(torch.int64)
        out = torch.zeros(num_seqs, max_kv, width, dtype=index_cache.dtype, device=device)
        for b in range(num_seqs):
            kl = int(kv_lens[b].item())
            if kl == 0:
                continue
            n_full = kl // block_size
            tail = kl - n_full * block_size
            rows = []
            blk = bt[b]
            if n_full > 0:
                slot_ids = (blk[:n_full, None] * block_size + torch.arange(block_size, device=device)[None, :]).reshape(-1)
                rows.append(flat.index_select(0, slot_ids))
            if tail > 0:
                last_blk = int(blk[n_full].item())
                slot_ids = last_blk * block_size + torch.arange(tail, device=device)
                rows.append(flat.index_select(0, slot_ids))
            packed = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
            out[b, :kl] = packed
        return out

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        """KDA delta-rule over framework conv/ssm slots.

        Returns ``[B, S, num_heads_local, head_dim]``. The conv1d + delta-rule
        math is identical to the model-layer self-contained path (validated
        against the transformers reference); only the state I/O moved here so
        both attention layer types dispatch through the backend.
        """
        from xllm.python.models.glm5_next import (
            _causal_conv1d_fn,
            _causal_conv1d_update,
            _l2norm,
        )
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        layer_cache = self._kv_caches[layer.layer_id]
        conv_cache = layer_cache.conv
        ssm_cache = layer_cache.ssm
        assert conv_cache is not None and ssm_cache is not None, (
            "execute_linear requires a linear-attention layer cache (conv/ssm)")

        batch_size, _, seq_len = mixed_qkv.shape
        conv_kernel_size = layer.conv_kernel_size
        conv_state_len = conv_kernel_size - 1
        head_dim = layer.head_dim
        num_heads_local = layer.num_heads_local
        qkv_dim = layer.qkv_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        # Raw (pre-conv) projections for the lazy-commit stash, kept alive
        # across the branches below (mixed_qkv is reassigned post-conv), and
        # the per-seq read-only chain bookkeeping for plain steps (see the
        # non-verify branch).
        raw_mixed, raw_gate, raw_beta = mixed_qkv, gate, beta
        chain_seqs: dict = {}

        idx = metadata.linear_state_indices
        num_seqs = idx.shape[0] if idx is not None else batch_size
        # ACL-graph decode with a flattened batch: the model forward unsqueezes
        # the 1-D ``[num_seqs]`` decode input into ``[1, num_seqs]``, so
        # mixed_qkv arrives as ``[1, conv_dim, num_seqs]`` with idx
        # ``[num_seqs]``. The eager multi-sequence branch (a Python loop over
        # q_cu_seq_lens with host syncs) is not graph-capturable; decode is
        # exactly one token per sequence, so reshape to ``[num_seqs, conv_dim,
        # 1]`` and take the simple per-sequence path with static shapes. gate
        # ``[1, T, nh, hd]`` / beta ``[1, T, nh]`` follow the same transpose.
        in_graph = _in_acl_graph()
        is_decode = not metadata.is_prefill and not metadata.is_chunked_prefill
        flatten_graph_decode = (
            in_graph
            and is_decode
            and idx is not None
            and batch_size == 1
            and num_seqs > 1
            and seq_len == num_seqs
        )
        if flatten_graph_decode:
            mixed_qkv = mixed_qkv.transpose(0, 2)  # [1, C, T] -> [T, C, 1]
            batch_size, _, seq_len = mixed_qkv.shape
            hidden_shape = (batch_size, seq_len, -1, head_dim)
            gate = gate.transpose(0, 1)  # [1, T, nh, hd] -> [T, 1, nh, hd]
            beta = beta.transpose(0, 1)  # [1, T, nh] -> [T, 1, nh]
        if idx is None:
            device = mixed_qkv.device
            conv_state = torch.zeros(
                batch_size, layer.conv_dim, conv_state_len,
                dtype=layer.conv1d.weight.dtype, device=device,
            )
            ssm_state = torch.zeros(
                batch_size, num_heads_local, head_dim, head_dim,
                dtype=torch.float32, device=device,
            )
        else:
            if idx.dtype != torch.int64:
                idx = idx.to(torch.int64)
            # MTP spec-verify expands one logical sequence into consecutive
            # batch rows (bonus + drafted tokens, e.g. q_cu=[0,1,2,3,4] for
            # two k=1 sequences, kv=[n,n+1,m,m+1] per row) while
            # linear_state_indices stays per SEQUENCE (one slot id per
            # sequence, e.g. [s1,s2] for the four rows above). Expand each
            # sequence's slot across its contiguous row group, then merge the
            # same-slot rows back into single sequences: the recurrent state
            # must chain bonus -> draft inside one sequence, and the cache
            # read/write needs one row per slot. Mapping rows to sequences by
            # position instead (rows==seqs) silently drops the tail rows of
            # every sequence past the first and corrupts the view/layout
            # downstream.
            merged_q_cu: Optional[torch.Tensor] = None
            merged_row0: list = []
            q_cu_raw = metadata.q_cu_seq_lens
            # The row-merge + lazy-commit bookkeeping below is MTP-spec-verify
            # tracking that relies on device->host syncs (.item()/.tolist()),
            # which are forbidden on a captured ACL-graph stream. The graph
            # decode path is not spec-verify (merged_q_cu stays None and the
            # simple path below uses the capture-safe conv), so skip it whole.
            q_rows = (0 if in_graph else
                      (int(q_cu_raw.numel()) - 1
                       if q_cu_raw is not None else 0))
            per_row_idx = None
            if q_rows == idx.numel() and idx.numel() > 1 and bool(
                    (idx[1:] == idx[:-1]).any().item()):
                # Defensive: indices already duplicated per row.
                per_row_idx = idx
            elif q_rows > idx.numel() and q_rows % idx.numel() == 0:
                # Spec-verify expansion (uniform rows per sequence = k+1).
                per_row_idx = idx.repeat_interleave(q_rows // idx.numel())
            elif q_rows > idx.numel():
                raise RuntimeError(
                    f"unaligned linear-state batch: {q_rows} rows vs "
                    f"{idx.numel()} sequences with non-uniform expansion")
            if per_row_idx is not None:
                row_lengths = (q_cu_raw[1:] - q_cu_raw[:-1]).tolist()
                merged_lengths: list = []
                merged_slots: list = []
                for row, (slot, length) in enumerate(
                        zip(per_row_idx.tolist(), row_lengths)):
                    if merged_slots and slot == merged_slots[-1]:
                        merged_lengths[-1] += length
                    else:
                        merged_slots.append(slot)
                        merged_lengths.append(length)
                        merged_row0.append(row)
                merged_cu = [0]
                for length in merged_lengths:
                    merged_cu.append(merged_cu[-1] + length)
                merged_q_cu = torch.tensor(
                    merged_cu, dtype=torch.int64, device=idx.device)
                idx = torch.tensor(
                    merged_slots, dtype=torch.int64, device=idx.device)
                num_seqs = idx.shape[0]
            else:
                # Non-verify path (plain decode / prefill): states are
                # committed wholesale the regular way, plus lazy-commit
                # handshakes - but ONLY once this backend has seen a
                # spec-verify batch, so a pure-decode engine (no MTP) keeps
                # its exact pre-MTP behavior. (1) A reject interrupts the
                # verify chain with a bootstrap plain step while the chain's
                # last confirmed token is still stashed un-advanced - consume
                # that stash FIRST (advance the live caches), or the token is
                # silently lost from the linear state at every rejection.
                # (2) A plain step with an open chain is processed
                # READ-ONLY: its row is stashed and committed by the next
                # verify's advance, so both its own output and the re-
                # processed row0 in the next verify match a plain decode
                # exactly (committing here would make the next verify's row0
                # run from a state already containing it). (3) Other plain
                # steps record a coverage marker (kv length, empty stash) so
                # the next verify does not re-advance them. Prefill resets
                # the chain (the slot restarts from a fresh state).
                if not _MTP_FULL_COMMIT and not in_graph:
                    _pending = getattr(self, "_mtp_pending", None)
                    if _pending is None:
                        _pending = {}
                        self._mtp_pending = _pending
                    _lp = _pending.setdefault(layer.layer_id, {})
                    # NOTE: kv_seq_lens.tolist() / idx.tolist() are device->host
                    # syncs forbidden on a captured stream; the graph path uses
                    # static metadata and the capture-safe conv in the simple
                    # path below, so this lazy-commit bookkeeping is eager-only.
                    _kv_list = (metadata.kv_seq_lens.tolist()
                                if metadata.kv_seq_lens is not None else None)
                    _active = getattr(self, "_mtp_seen_verify", False)
                    _is_pf = (metadata.is_prefill
                              or metadata.is_chunked_prefill)
                    if (_MTP_TRACE and layer.layer_id == 0 and not _is_pf
                            and getattr(self, "_mtp_stats", None) is not None):
                        # Non-prefill plain steps inside the decode phase are
                        # the rejection bootstraps - their share of all steps
                        # is the acceptance-rate signal.
                        self._mtp_stats["plain"] += num_seqs
                    _cw = layer.conv1d.weight.squeeze(1)
                    for s, slot in enumerate(idx.tolist()):
                        slot = int(slot)
                        if _is_pf or _kv_list is None or s >= len(_kv_list):
                            _lp.pop(slot, None)
                            continue
                        prev = _lp.get(slot)
                        if not _active or prev is None:
                            _lp[slot] = (
                                int(_kv_list[s]), mixed_qkv[:, :, 0:0],
                                gate[:, 0:0], beta[:, 0:0],
                                _cw, layer.activation)
                            continue
                        (prev_base, prev_seg, prev_g, prev_b,
                         _pw, _pact) = prev
                        _m = int(_kv_list[s]) - 1 - prev_base
                        if 0 < _m <= prev_seg.shape[2]:
                            _cs = (conv_cache[slot]
                                   .transpose(0, 1).unsqueeze(0)
                                   .contiguous())
                            _cin = torch.cat(
                                [_cs, prev_seg[:, :, :_m]], dim=-1)
                            _adv_qkv = _causal_conv1d_fn(
                                _cin, _cw, layer.activation,
                            )[:, :, -_m:]
                            conv_cache[slot] = _cin[
                                0, :, -conv_state_len:].transpose(0, 1)
                            _sp = torch.split(
                                _adv_qkv.transpose(1, 2),
                                [qkv_dim] * 3, dim=-1)
                            _, _st = recurrent_kda(
                                _sp[0].reshape(
                                    -1, num_heads_local, head_dim
                                ).to(torch.bfloat16).contiguous(),
                                _sp[1].reshape(
                                    -1, num_heads_local, head_dim
                                ).to(torch.bfloat16).contiguous(),
                                _sp[2].reshape(
                                    -1, num_heads_local, head_dim
                                ).to(torch.bfloat16).contiguous(),
                                prev_g[:, :_m].reshape(
                                    -1, num_heads_local, head_dim
                                ).to(torch.float32).contiguous(),
                                prev_b[:, :_m].reshape(
                                    -1, num_heads_local
                                ).to(torch.float32).contiguous(),
                                initial_state=ssm_cache[slot].unsqueeze(0),
                                cu_seqlens=torch.tensor(
                                    [0, _m], dtype=torch.int32,
                                    device=mixed_qkv.device),
                                layout="TND",
                                scale=1.0 / (head_dim ** 0.5),
                                output_final_state=True,
                                inplace_final_state=False,
                                use_qk_l2norm_in_kernel=True,
                                use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False,
                                state_v_first=True,
                            )
                            ssm_cache[slot] = _st[0].to(ssm_cache.dtype)
                        if 0 <= _m <= prev_seg.shape[2]:
                            # Open chain: read-only plain; stash at the
                            # tail write-back.
                            _lp.pop(slot, None)
                            chain_seqs[s] = slot
                            continue
                        _lp[slot] = (
                            int(_kv_list[s]), mixed_qkv[:, :, 0:0],
                            gate[:, 0:0], beta[:, 0:0],
                            _cw, layer.activation)
            if (not _MTP_FULL_COMMIT and merged_q_cu is not None
                    and not in_graph
                    and getattr(self, "_mtp_pending", None)
                    and layer.layer_id == min(self._mtp_pending.keys())):
                # Batched cross-layer lazy advance. Every KDA layer's
                # verify-step advance is independent (own weights/states,
                # same confirmed-token count), so the FIRST KDA layer of the
                # step advances them ALL here - one conv per layer plus ONE
                # recurrent_kda for the whole batch - instead of one extra
                # recurrent call inside each layer (~33 extra kernel
                # launches per step otherwise). The coordinator runs before
                # any layer reads the caches, so every layer (including this
                # one) reads its already-advanced state below. Non-consumable
                # entries (coverage markers, m == 0 open chains) are left in
                # place for the per-layer logic.
                _slot_base = {}
                _kv_all = (metadata.kv_seq_lens.tolist()
                           if metadata.kv_seq_lens is not None else None)
                if _kv_all is not None:
                    for _si, _sl in enumerate(idx.tolist()):
                        _slot_base.setdefault(
                            int(_sl), _kv_all[merged_row0[_si]] - 1)
                _bq, _bk, _bv, _bg, _bb = [], [], [], [], []
                _b_init, _b_scatter = [], []
                _b_cu = [0]
                for _lid, _lp2 in self._mtp_pending.items():
                    _lc = self._kv_caches[_lid]
                    for _sl, (_pb, _seg, _pg, _pbeta,
                              _pw, _pact) in list(_lp2.items()):
                        _base = _slot_base.get(_sl)
                        if _base is None:
                            continue
                        _bm = _base - _pb
                        if not (0 < _bm <= _seg.shape[2]):
                            continue
                        del _lp2[_sl]
                        _cs = (_lc.conv[_sl]
                               .transpose(0, 1).unsqueeze(0).contiguous())
                        _cin = torch.cat([_cs, _seg[:, :, :_bm]], dim=-1)
                        _cout = _causal_conv1d_fn(
                            _cin, _pw, _pact)[:, :, -_bm:]
                        _lc.conv[_sl] = _cin[
                            0, :, -conv_state_len:].transpose(0, 1)
                        _sp = torch.split(
                            _cout.transpose(1, 2), [qkv_dim] * 3, dim=-1)
                        _bq.append(_sp[0].reshape(
                            -1, num_heads_local, head_dim
                        ).to(torch.bfloat16))
                        _bk.append(_sp[1].reshape(
                            -1, num_heads_local, head_dim
                        ).to(torch.bfloat16))
                        _bv.append(_sp[2].reshape(
                            -1, num_heads_local, head_dim
                        ).to(torch.bfloat16))
                        _bg.append(_pg[:, :_bm].reshape(
                            -1, num_heads_local, head_dim).to(torch.float32))
                        _bb.append(_pbeta[:, :_bm].reshape(
                            -1, num_heads_local).to(torch.float32))
                        _b_init.append(_lc.ssm[_sl].unsqueeze(0))
                        _b_scatter.append((_lid, _sl))
                        _b_cu.append(_b_cu[-1] + _bm)
                if _bq:
                    _, _bst = recurrent_kda(
                        torch.cat(_bq).contiguous(),
                        torch.cat(_bk).contiguous(),
                        torch.cat(_bv).contiguous(),
                        torch.cat(_bg).contiguous(),
                        torch.cat(_bb).contiguous(),
                        initial_state=torch.cat(_b_init, dim=0).contiguous(),
                        cu_seqlens=torch.tensor(
                            _b_cu, dtype=torch.int32,
                            device=mixed_qkv.device),
                        layout="TND", scale=1.0 / (head_dim ** 0.5),
                        output_final_state=True, inplace_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False,
                        state_v_first=True,
                    )
                    for _bi, (_lid, _sl) in enumerate(_b_scatter):
                        self._kv_caches[_lid].ssm[_sl] = _bst[_bi].to(
                            self._kv_caches[_lid].ssm.dtype)
            conv_i = conv_cache.index_select(0, idx)
            conv_i = conv_i.transpose(1, 2).contiguous()
            ssm_i = ssm_cache.index_select(0, idx)
            his = metadata.has_initial_state
            if his is not None and len(his) == num_seqs:
                if not isinstance(his, torch.Tensor):
                    his = torch.tensor(
                        his, dtype=torch.int64, device=conv_i.device
                    )
                warm = his.to(torch.bool).view(num_seqs, 1, 1)
                conv_i = torch.where(warm, conv_i, torch.zeros_like(conv_i))
                ssm_i = torch.where(
                    warm.view(num_seqs, 1, 1, 1),
                    ssm_i,
                    torch.zeros_like(ssm_i),
                )
            conv_state, ssm_state = conv_i, ssm_i.contiguous()
            if os.environ.get("GLM5NEXT_DEBUG_LINEAR") and not (
                    get_forward_context_or_none() is not None
                    and get_forward_context_or_none().acl_graph is not None
                ):
                print(
                    f"[linear-debug-in] L{layer.layer_id} graph={in_graph} "
                    f"conv_in={conv_state.abs().sum().item():.6e} "
                    f"ssm_in={ssm_state.abs().sum().item():.6e} "
                    f"mqkv_in={mixed_qkv.abs().sum().item():.6e} "
                    f"gate_in={gate.abs().sum().item():.6e} "
                    f"beta_in={beta.abs().sum().item():.6e}",
                    flush=True,
                )
            if chain_seqs:
                # Entry snapshots (post consume-advance) for the read-only
                # plain chain rows: restored at the tail write-back so this
                # step's own token is not committed here.
                _entry_conv = {i: conv_state[i].clone() for i in chain_seqs}
                _entry_ssm = {i: ssm_state[i].clone() for i in chain_seqs}

        conv_weight = layer.conv1d.weight.squeeze(1)
        activation = layer.activation
        scale = 1.0 / (head_dim ** 0.5)
        # Route on metadata, not seq_len: MTP/spec decode can carry multiple
        # tokens per sequence (seq_len > 1) but is still a decode step; the
        # seq_len heuristic would wrongly send it to the chunked prefill path.
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        device = mixed_qkv.device
        # Spec-verify (merged same-slot rows) commits lazily via the
        # kv-delta scheme in the flattened branch; see the comments there.
        commit_first_only = False
        # A merged spec-verify batch must take the flattened branch even when
        # it collapses to ONE sequence (single-stream verify: 2 rows -> 1
        # merged seq == batch_size): the simple path full-commits both rows,
        # which inserts a phantom draft token into the state on every
        # rejection (the corrected token overwrites the draft's position, so
        # only the bonus row of the previous step is confirmable here - see
        # the lazy-commit scheme below).
        if num_seqs == batch_size and merged_q_cu is None:
            # Simple path: mixed_qkv is already [B, conv_dim, S] (one sequence
            # per batch row, or a single flattened sequence).
            if seq_len == 1:
                if in_graph:
                    # F.conv1d is an aclop NPUGraph cannot capture; the manual
                    # depthwise mul-add is capture-safe. Eager keeps F.conv1d.
                    from xllm.python.models.glm5_next import (
                        _causal_conv1d_update_graph,
                    )
                    mixed_qkv = _causal_conv1d_update_graph(
                        mixed_qkv, conv_state, conv_weight, activation
                    )
                else:
                    mixed_qkv = _causal_conv1d_update(
                        mixed_qkv, conv_state, conv_weight, activation
                    )
            else:
                conv_in = torch.cat([conv_state, mixed_qkv], dim=-1)
                mixed_qkv = _causal_conv1d_fn(
                    conv_in, conv_weight, activation
                )[:, :, -seq_len:]
                conv_state = conv_in[..., -conv_state_len:]
                if conv_state.shape[-1] < conv_state_len:
                    conv_state = F.pad(
                        conv_state,
                        (conv_state_len - conv_state.shape[-1], 0), value=0,
                    )
        else:
            # Flattened multi-sequence: mixed_qkv is [1, conv_dim, T] (T = sum
            # of per-seq token counts). Variable-length (MTP/spec decode: each
            # sequence may carry a different token count) is supported by a
            # per-sequence conv1d loop (pure-torch F.conv1d is batched and
            # requires equal lengths) followed by a single varlen recurrent_kda
            # call (cu_seqlens does the per-seq split inside the kernel).
            q_cu = merged_q_cu if merged_q_cu is not None else (
                metadata.q_cu_seq_lens)
            assert q_cu is not None, (
                "multi-sequence linear attention needs q_cu_seq_lens")
            q_cu = q_cu.to(torch.int64)
            q_cu_list = q_cu.tolist()
            commit_first_only = merged_q_cu is not None
            if commit_first_only:
                # First spec-verify batch arms the plain-step handshakes (a
                # pure-decode backend never sets this and keeps the exact
                # pre-MTP plain-path behavior).
                self._mtp_seen_verify = True
                if _MTP_TRACE and layer.layer_id == 0:
                    _st = getattr(self, "_mtp_stats", None)
                    if _st is None:
                        _st = {"verify": 0, "plain": 0}
                        self._mtp_stats = _st
                    _st["verify"] += num_seqs
                    if _st["verify"] >= 50:
                        _mh = getattr(self, "_mtp_mhist", {})
                        _mh_str = " ".join(
                            f"m{k}={v}" for k, v in sorted(_mh.items()))
                        with open("/tmp/mtp_accept.log", "a") as _fh:
                            _fh.write(
                                f"[accept] pid={os.getpid()} "
                                f"verify={_st['verify']} "
                                f"plain(reject)={_st['plain']} "
                                f"accept_rate="
                                f"{_st['verify'] / max(1, _st['verify'] + _st['plain']):.2f} "
                                f"{_mh_str}\n")
                        _st["verify"] = 0
                        _st["plain"] = 0
                        self._mtp_mhist = {}
                # ---- MTP spec-verify steps (observed layout, k=1) ----
                # Rows are [last-confirmed token (re-processed each step to
                # judge the next draft), drafted token]; the C++ commits
                # exactly one token per step (the draft on accept, the
                # corrected argmax on reject) and the next step's row0 is
                # that token at the previous row1's position, so kv grows
                # +1 per step. Alternative scheme (full,
                # GLM5_MTP_COMMIT=full): commit BOTH rows every step - the
                # row0 re-write is near-idempotent under the delta rule but
                # a rejected draft leaves a persistent phantom write.
                #
                # Default scheme (lazy, see _MTP_FULL_COMMIT): commit
                # NOTHING here; (1) ADVANCE the live state by the PREVIOUS
                # step's confirmed tokens - their count m (kv_seq_lens
                # growth over the previous verify) with the stashed raw
                # qkv/gate rows; (2) process the current rows read-only
                # (outputs only); (3) stash the current rows for the next
                # step's advance. The state then tracks the true token
                # stream exactly, independent of accept/reject.
                kv_rows = metadata.kv_seq_lens.tolist()
                pending = getattr(self, "_mtp_pending", None)
                if pending is None:
                    pending = {}
                    self._mtp_pending = pending
                # The stash is PER LAYER: all KDA layers of a model share this
                # backend instance, and each layer's stashed qkv/gate rows are
                # only valid for that layer's own conv/recurrent advance.
                # Keying by slot alone made every layer overwrite the others'
                # stashes (layer N's advance consumed the last layer's rows -
                # total state corruption under multi-layer models).
                pending = pending.setdefault(layer.layer_id, {})
                live_slots = {int(x) for x in idx.tolist()}
                for slot in list(pending.keys()):
                    if slot not in live_slots:
                        del pending[slot]
                gate_raw = gate.view(1, -1, -1) if gate.dim() == 2 else gate
                beta_raw = beta.view(1, -1, -1) if beta.dim() == 2 else beta
                adv_seg: list = []       # raw qkv [1, conv_dim, m]
                adv_g: list = []
                adv_b: list = []
                adv_seq: list = []       # merged-seq index per segment
                for s in range(num_seqs):
                    slot = int(idx[s])
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    base_now = int(kv_rows[merged_row0[s]]) - 1
                    prev = pending.get(slot)
                    lead = 0
                    if prev is not None and not _MTP_FULL_COMMIT:
                        (prev_base, prev_seg, prev_g, prev_b,
                         _pw, _pact) = prev
                        m = base_now - prev_base
                        if _MTP_TRACE and layer.layer_id == 0:
                            _mh = getattr(self, "_mtp_mhist", None)
                            if _mh is None:
                                _mh = {}
                                self._mtp_mhist = _mh
                            _mh[m] = _mh.get(m, 0) + 1
                        if 0 < m <= prev_seg.shape[2]:
                            adv_seg.append(prev_seg[:, :, :m])
                            adv_g.append(prev_g[:, :m])
                            adv_b.append(prev_b[:, :m])
                            adv_seq.append(s)
                        # Rows already covered by the recorded coverage
                        # (e.g. the plain step before the first verify) must
                        # not be stashed again - trim them off the front.
                        lead = max(0, min(prev_base - base_now, t1 - t0))
                    if not _MTP_FULL_COMMIT:
                        pending[slot] = (
                            base_now + lead,
                            mixed_qkv[:, :, t0 + lead:t1].clone(),
                            gate_raw[:, t0 + lead:t1].clone(),
                            beta_raw[:, t0 + lead:t1].clone(),
                            conv_weight, activation,
                        )
                if adv_seg:
                    # conv-advance each segment sequentially (per slot) and
                    # collect post-conv qkv for the recurrent advance.
                    adv_conv_out = []
                    for s, seg_raw in zip(adv_seq, adv_seg):
                        cs = conv_state[s:s + 1].contiguous()
                        seg_raw = seg_raw.contiguous()
                        cin = torch.cat([cs, seg_raw], dim=-1)
                        adv_conv_out.append(_causal_conv1d_fn(
                            cin, conv_weight, activation
                        )[:, :, -seg_raw.shape[2]:])
                        conv_state[s] = cin[0, :, -conv_state_len:]
                    adv_qkv = torch.cat(adv_conv_out, dim=-1)
                    a_lengths = [seg.shape[2] for seg in adv_seg]
                    a_cu = [0]
                    for length in a_lengths:
                        a_cu.append(a_cu[-1] + length)
                    a_total = a_cu[-1]
                    a_split = torch.split(
                        adv_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
                    aq = a_split[0].reshape(-1, num_heads_local,
                                            head_dim).to(torch.bfloat16).contiguous()
                    ak = a_split[1].reshape(-1, num_heads_local,
                                            head_dim).to(torch.bfloat16).contiguous()
                    av = a_split[2].reshape(-1, num_heads_local,
                                            head_dim).to(torch.bfloat16).contiguous()
                    ag = torch.cat(adv_g, dim=1).reshape(
                        -1, num_heads_local, head_dim).to(torch.float32).contiguous()
                    ab = torch.cat(adv_b, dim=1).reshape(
                        -1, num_heads_local).to(torch.float32).contiguous()
                    adv_idx = torch.tensor(adv_seq, dtype=torch.long,
                                           device=ssm_state.device)
                    if _KDA_SEQWISE and len(adv_seq) > 1:
                        # Same seq-wise rationale as the read-only branch: keep
                        # every advance call at the single-sequence shape so
                        # concurrent batches advance state bit-identically to a
                        # single-request run.
                        for _si, _slot in enumerate(adv_seq):
                            _a0, _a1 = a_cu[_si], a_cu[_si + 1]
                            _asel = slice(int(_a0), int(_a1))
                            _, _st = recurrent_kda(
                                aq[_asel].contiguous(), ak[_asel].contiguous(),
                                av[_asel].contiguous(), ag[_asel].contiguous(),
                                ab[_asel].contiguous(),
                                initial_state=ssm_state.index_select(
                                    0, adv_idx[[_si]]),
                                cu_seqlens=torch.tensor(
                                    [0, int(_a1 - _a0)], dtype=torch.int32,
                                    device=device),
                                layout="TND", scale=scale,
                                output_final_state=True,
                                inplace_final_state=False,
                                use_qk_l2norm_in_kernel=True,
                                use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False,
                                state_v_first=True,
                            )
                            ssm_state[adv_idx[_si]] = _st.to(ssm_state.dtype)
                    else:
                        _, adv_state = recurrent_kda(
                            aq, ak, av, ag, ab,
                            initial_state=ssm_state.index_select(0, adv_idx),
                            cu_seqlens=torch.tensor(
                                a_cu, dtype=torch.int32, device=device),
                            layout="TND", scale=scale,
                            output_final_state=True, inplace_final_state=False,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=False,
                            use_beta_sigmoid_in_kernel=False,
                            state_v_first=True,
                        )
                        ssm_state[adv_idx] = adv_state.to(ssm_state.dtype)
                outs = []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    seg = mixed_qkv[:, :, t0:t1].contiguous()
                    # contiguous: a row-view of the batched cache has a
                    # different layout than a single-request's whole cache
                    # and the conv kernel picks its tiling off that layout.
                    cs = conv_state[s:s + 1].contiguous()
                    cin = torch.cat([cs, seg], dim=-1)
                    outs.append(_causal_conv1d_fn(
                        cin, conv_weight, activation,
                    )[:, :, -(t1 - t0):])
                    if _MTP_FULL_COMMIT:
                        # Full-commit scheme keeps the live conv state at the
                        # tail of BOTH rows (mirrors the pre-merge simple
                        # path); lazy mode leaves it at the advance boundary.
                        conv_state[s] = cin[0, :, -conv_state_len:]
                mixed_qkv = torch.cat(outs, dim=-1)
            else:
                outs = []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    seg = mixed_qkv[:, :, t0:t1]   # [1, conv_dim, seg_len]
                    cs = conv_state[s:s + 1]       # [1, conv_dim, state_len]
                    seg_len = t1 - t0
                    if seg_len == 1:
                        outs.append(_causal_conv1d_update(
                            seg, cs, conv_weight, activation
                        ))
                    else:
                        cin = torch.cat([cs, seg], dim=-1)
                        outs.append(_causal_conv1d_fn(
                            cin, conv_weight, activation
                        )[:, :, -seg_len:])
                        conv_state[s] = cin[0, :, -conv_state_len:]
                        if conv_state.shape[-1] < conv_state_len:
                            conv_state[s] = F.pad(
                                conv_state[s],
                                (conv_state_len - conv_state.shape[-1], 0),
                                value=0,
                            )
                mixed_qkv = torch.cat(outs, dim=-1)   # [1, conv_dim, T]
            # TND packed layout for recurrent_kda: [T, nh, hd] per channel group.
            seq_len = int(q_cu_list[-1])
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1
        )
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        g = gate if num_seqs == batch_size else gate.view(hidden_shape)
        # ``beta`` arrives as [B, S, nh] and is already correct for both
        # layouts: per-sequence [num_seqs, per_seq_len, nh] when the batch rows
        # map 1:1 to sequences, and flattened [1, T, nh] (T = sum of q_cu) for
        # the varlen multi-sequence path. Re-viewing it as
        # (num_seqs, seq_len, nh) assumes a uniform per-seq length == the
        # flattened total and crashes on multi-sequence decode batches
        # (e.g. 2 concurrent requests: view [2, 2, 4] on 8 elements).
        b = beta
        # fla_npu KDA ops require fp32 gate/beta (the pure-torch reference also
        # upcasts them); the model hands them in bf16.
        g = g.to(torch.float32)
        b = b.to(torch.float32)
        if not is_prefill:
            # decode (incl. MTP multi-token-per-seq varlen): recurrent_kda on
            # packed TND [T, nh, hd] with cu_seqlens.
            q_tnd = query.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            k_tnd = key.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            v_tnd = value.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            g_tnd = g.reshape(-1, num_heads_local, head_dim).contiguous()
            b_tnd = b.reshape(-1, num_heads_local).contiguous()
            if num_seqs != batch_size:
                cu_seqlens = q_cu.to(torch.int32)
            elif seq_len == 1:
                # B independent single-token sequences.
                if in_graph:
                    # Constant content; allocate once into the persistent
                    # execution buffer so capture records no per-step H2D
                    # arange.
                    cu_seqlens = get_execution_buffer(
                        ("KDA_DECODE_CU_SEQLENS", num_seqs),
                        lambda: torch.arange(
                            num_seqs + 1, dtype=torch.int32, device=device
                        ),
                    )
                else:
                    cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
            else:
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            if commit_first_only and not is_prefill and not _MTP_FULL_COMMIT:
                # Spec-verify: the state was already ADVANCED by the
                # previous step's confirmed tokens (see the flattened
                # branch). Chain the current rows read-only from that state
                # for their outputs; the tail write-back persists exactly
                # the advanced state (never the drafted rows).
                # state_v_first=True matches the V-first accumulation order of
                # the KDA reference and the prefill/advance paths so every
                # read here is layout-consistent with the advanced state.
                # inplace_final_state stays False on purpose: the lazy-advance
                # scheme owns ssm_state mutations (advance step writes, verify
                # reads); an in-place write here would clobber the advanced
                # state before the tail write-back persists it.
                if _KDA_SEQWISE and num_seqs > 1 and num_seqs != batch_size:
                    # Seq-wise dispatch: one single-segment recurrent call per
                    # sequence, exactly matching the single-request verify call
                    # shape. The flattened multi-segment call picks a different
                    # device tiling inside the engine process (verified
                    # bit-exact offline but not in-process), leaking a 1-ULP
                    # drift into the KDA output that the next layer's mHC
                    # sinkhorn amplifies ~16x per layer until argmax flips —
                    # concurrent outputs then diverge from single-request ones.
                    _ros = []
                    for s in range(num_seqs):
                        _t0, _t1 = q_cu_list[s], q_cu_list[s + 1]
                        _sel = slice(int(_t0), int(_t1))
                        _ro = recurrent_kda(
                            q_tnd[_sel].contiguous(), k_tnd[_sel].contiguous(),
                            v_tnd[_sel].contiguous(), g_tnd[_sel].contiguous(),
                            b_tnd[_sel].contiguous(),
                            initial_state=ssm_state[s:s + 1].contiguous(),
                            cu_seqlens=torch.tensor(
                                [0, int(_t1 - _t0)], dtype=torch.int32,
                                device=device),
                            layout="TND", scale=scale,
                            output_final_state=False, inplace_final_state=False,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=False,
                            use_beta_sigmoid_in_kernel=False,
                            state_v_first=True,
                        )
                        _ros.append(_ro[0] if isinstance(_ro, tuple) else _ro)
                    core_attn_out = torch.cat(_ros, dim=0)
                    final_state = ssm_state
                else:
                    ro = recurrent_kda(
                        q_tnd, k_tnd, v_tnd, g_tnd, b_tnd,
                        initial_state=ssm_state,
                        cu_seqlens=cu_seqlens,
                        layout="TND", scale=scale,
                        output_final_state=False, inplace_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False,
                        state_v_first=True,
                    )
                    # fla_npu returns a tuple even with
                    # output_final_state=False.
                    core_attn_out = ro[0] if isinstance(ro, tuple) else ro
                    final_state = ssm_state
            else:
                core_attn_out, final_state = recurrent_kda(
                    q_tnd, k_tnd, v_tnd, g_tnd, b_tnd,
                    initial_state=ssm_state,
                    cu_seqlens=cu_seqlens,
                    layout="TND", scale=scale,
                    output_final_state=True, inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False,
                    state_v_first=True,
                )
            # recurrent_kda returns the packed TND [T, nh, hd] layout of its
            # inputs; restore the [B, S, nh, hd] grouping the model layer
            # expects (o_norm gates per head). For T == 1 the flat layout
            # happens to broadcast identically, which masked this for
            # single-stream decode; a multi-sequence decode batch (2
            # concurrent requests flattened to [1, 2, ...]) surfaced it.
            core_attn_out = core_attn_out.to(query.dtype).reshape(hidden_shape)
        else:
            q_in = _l2norm(query.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            k_in = _l2norm(key.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            v_in = value.to(torch.bfloat16).contiguous()
            cu_seqlens = q_cu.to(torch.int32) if num_seqs != batch_size else \
                torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            if _KDA_SEQWISE and cu_seqlens.numel() > 2:
                # Seq-wise prefill: one single-sequence chunk_kda_fwd per
                # sequence. A merged multi-sequence prefill (engine batches the
                # concurrent requests' prefills) leaves per-seq conv/ssm state
                # that differs from a single-request prefill (state-fingerprint
                # verified: L0 state matches, L1+ diverges on seq1/seq2), and
                # that state drift propagates through every later verify step.
                # state_v_first=True pins the [HV,V,K] state layout so the
                # ssm state handed to decode matches the recurrent path
                # (default False is [HV,K,V] — K/V-transposed; K=V=128 hides
                # the shape mismatch while corrupting decode precision).
                _pout, _pstates = [], []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    sel = slice(int(t0), int(t1))
                    _r = chunk_kda_fwd(
                        q_in[:, sel].contiguous(), k_in[:, sel].contiguous(),
                        v_in[:, sel].contiguous(),
                        g[:, sel].contiguous(),
                        b[:, sel].contiguous(),
                        scale,
                        chunk_size=64, layout="BSND",
                        initial_state=ssm_state[s:s + 1],
                        output_final_state=True,
                        cu_seqlens=torch.tensor(
                            [0, int(t1 - t0)], dtype=torch.int32, device=device),
                        use_gate_in_kernel=False,
                        return_intermediate_states=False,
                        state_v_first=True,
                    )
                    _pout.append(_r[0])
                    _pstates.append(_r[1])
                core_attn_out = torch.cat(_pout, dim=0).to(query.dtype)
                final_state = torch.cat(_pstates, dim=0)
            else:
                result = chunk_kda_fwd(
                    q_in, k_in, v_in, g, b, scale,
                    chunk_size=64, layout="BSND",
                    initial_state=ssm_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_gate_in_kernel=False,
                    return_intermediate_states=False,
                    state_v_first=True,
                )
                core_attn_out = result[0].to(query.dtype)
                final_state = result[1]

        if idx is not None:
            if chain_seqs:
                # Read-only plain chain rows: restore the entry (post
                # consume-advance) state and stash this step's own raw rows;
                # the next verify's advance commits them exactly once.
                _pend = self._mtp_pending[layer.layer_id]
                _kv_tail = (metadata.kv_seq_lens.tolist()
                            if metadata.kv_seq_lens is not None else None)
                for i, slot in chain_seqs.items():
                    conv_state[i] = _entry_conv[i]
                    final_state[i] = _entry_ssm[i]
                    if num_seqs != batch_size:
                        t0, t1 = q_cu_list[i], q_cu_list[i + 1]
                        seg = raw_mixed[:, :, t0:t1]
                        gseg = raw_gate[:, t0:t1]
                        bseg = raw_beta[:, t0:t1]
                    else:
                        seg = raw_mixed[i:i + 1]
                        gseg = raw_gate[i:i + 1]
                        bseg = raw_beta[i:i + 1]
                    _pend[slot] = (
                        int(_kv_tail[i]) - seg.shape[2],
                        seg.clone(), gseg.clone(), bseg.clone(),
                        layer.conv1d.weight.squeeze(1), layer.activation)
            conv_cache.index_copy_(
                0, idx, conv_state.transpose(1, 2).contiguous()
            )
            ssm_cache.index_copy_(
                0, idx, final_state.float().contiguous()
            )
        if os.environ.get("GLM5NEXT_DEBUG_LINEAR") and not (
            get_forward_context_or_none() is not None
            and get_forward_context_or_none().acl_graph is not None
        ):
            print(
                f"[linear-debug] L{layer.layer_id} graph={in_graph} "
                f"bs={batch_size} sl={seq_len} ns={num_seqs} "
                f"idx={idx.flatten().tolist() if idx is not None else None} "
                f"his={metadata.has_initial_state.flatten().tolist() if isinstance(metadata.has_initial_state, torch.Tensor) else metadata.has_initial_state} "
                f"conv_sum={conv_state.abs().sum().item():.6e} "
                f"ssm_sum={ssm_state.abs().sum().item():.6e} "
                f"core_sum={core_attn_out.abs().sum().item():.6e}",
                flush=True,
            )
        # multi-seq path reshaped mixed_qkv to [num_seqs, ...]; flatten the
        # output back to [1, T, ...] so the KDA forward's hidden_shape [1, T]
        # aligns for o_norm / o_proj.
        if num_seqs != batch_size:
            core_attn_out = core_attn_out.reshape(1, -1, *core_attn_out.shape[2:])
        elif flatten_graph_decode:
            # The flatten-decode graph branch ran the simple path on
            # [num_seqs, 1, nh, hd]; restore the model's flattened [1, T, ...]
            # layout so o_norm's gate ([1, T, nh, hd]) aligns without
            # broadcasting.
            core_attn_out = core_attn_out.transpose(0, 1)
        return core_attn_out

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        out = get_execution_buffer(
            ("SFA_OUTPUT", layer_id) + tuple(q_latent.shape),
            lambda: torch.empty_like(q_latent),
        )
        return kernels.sparse_flash_attention_out(
            q_latent,
            nope_cache,
            nope_cache,
            topk,
            block_table,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
            q_pe,
            rope_cache,
            self.scale,
            1,
            "TND",
            "PA_BSND",
            3,
            out,
        )  # [T, H, kv_lora]

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self, q_3d: torch.Tensor, k_3d: torch.Tensor, v_3d: torch.Tensor,
        k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        # Prefix-cache hit (or chunked prefill with prior context): part of the
        # KV already lives in the paged cache, so this forward only carries the
        # new tokens (q_len < kv_len). Attend over the full paged KV via
        # block_table, mirroring _decode. Without this, the new query tokens
        # would only see their own KV (actual_seq_lengths_kv == q_len) and never
        # the cached prefix, diverging from a full recompute.
        if metadata.block_table is not None:
            block_size = k_cache.size(1)
            k_flat = k_cache.view(k_cache.size(0), block_size, -1)
            v_flat = v_cache.view(v_cache.size(0), block_size, -1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d, k_flat, v_flat,
                pse_shift=None,
                atten_mask=self._causal_mask,
                block_table=self._block_table_i32,
                actual_seq_lengths=actual_seq,
                actual_seq_lengths_kv=self._actual_seq_kv,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                block_size=block_size,
                sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
                softmax_lse_flag=False,
            )
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_3d, v_3d,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Context-Parallel prefill: all-gather KV, attend over causal prefix
    # ------------------------------------------------------------------

    def _prefill_cp(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: "CpContext",
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill attention for this rank's zigzag sequence shard.

        q/k/v hold this rank's ``total_local`` rows (two owned chunks per
        sequence, padding rows zeroed). We all-gather K/V back to the full
        global-order sequence, write the complete KV into the paged cache (so a
        later non-CP decode sees every position), then run one FIA over this
        rank's real queries. Each owned (sequence, half) segment is a packed
        sub-sequence: its ``real_count`` queries attend the causal prefix
        ``[0, segment_start + real_count)`` selected by ``kv_gather_index``.
        With ``sparse_mode=3`` (right-aligned causal) query row ``i`` of a
        segment attends KV ``[0, segment_start + i]`` — its exact global causal
        range. Segments are independent sub-sequences delimited by
        ``q_cu_seqlens`` / ``kv_cu_seqlens``, so both owned chunks resolve in a
        single call.
        """
        local_tokens = q_3d.shape[0]

        kv_global_k = cp_gather_kv(k_3d, cp_context)
        kv_global_v = cp_gather_kv(v_3d, cp_context)

        # Persist the full global-order KV into this rank's paged cache.
        kernels.reshape_paged_cache(
            metadata.slot_mapping,
            kv_global_k.contiguous(),
            kv_global_v.contiguous(),
            k_cache,
            v_cache,
        )

        # A CP rank can own only padding chunks when every sequence in the batch
        # is shorter than the zigzag chunk grid (e.g. a 1-token prompt with
        # cp_size > 1). It then has no real queries. The KV all-gather above
        # already ran (collectives must stay in lockstep across ranks) and the
        # full global KV is now in this rank's paged cache, so skip the FIA:
        # calling it with a 0-row query and empty actual_seq_lengths is rejected
        # by npu_fused_infer_attention. Return the all-zero shard directly.
        if cp_context.query_index.numel() == 0:
            return q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)

        # Real queries this rank owns, packed per (sequence, half) segment.
        q_real = q_3d.index_select(0, cp_context.query_index).contiguous()
        # Each segment's causal KV prefix, packed in the same segment order.
        kv_prefix_k = kv_global_k.index_select(0, cp_context.kv_gather_index).contiguous()
        kv_prefix_v = kv_global_v.index_select(0, cp_context.kv_gather_index).contiguous()

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_real,
            kv_prefix_k,
            kv_prefix_v,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=cp_context.q_cu_seqlens,
            actual_seq_lengths_kv=cp_context.kv_cu_seqlens,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        output = output.reshape(-1, self.num_heads * self.head_dim)

        # Scatter real-query outputs back into the padded [total_local] layout;
        # padding rows stay zero (they are never selected by restore_index in
        # the subsequent all-gather merge).
        out_local = q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)
        out_local.index_copy_(0, cp_context.query_index, output)
        return out_local

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_size: int,
    ) -> None:
        torch.ops.npu.npu_fused_infer_attention_score.out(
            q, k, v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=self._graph_workspace,
            out=[self._current_graph_output, self._current_graph_lse],
        )

    def _decode(
        self, q_3d: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(q_3d, k_flat, v_flat, block_size)
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(q_3d, k_flat, v_flat, block_size)

            graph_context.tasks.append(
                AclGraphTask(event, handle, _update_fia_args)
            )
            return self._current_graph_output.reshape(
                num_tokens, self.num_heads * self.head_dim
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_flat, v_flat,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q[:num_tokens],
            actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cumulative_seq_lens(
        self, metadata: AttentionMetadata, num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
