# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GLM-5-Next MTP (multi-token prediction) draft model.

The speculative worker and MTP scheduling remain in C++ (MTPWorkerImpl). This
module only describes the draft computation, mirroring the DeepSeek-V3.2
python MTP scheme (``deepseek_v32_mtp.py``):

- The draft layer is the checkpoint's appended layer
  ``model.language_model.layers.<n_layers>`` — a full DSA sparse-attention MoE
  layer WITHOUT mHC residual streams (its checkpoint keys carry
  ``input_layernorm`` / ``post_attention_layernorm`` and no ``hc_*``
  weights), so the decoder layer here is a plain pre-norm residual variant.
- ``eh_proj`` fuses ``[enorm(token_embed) ‖ hnorm(carried_hidden)]`` —
  embedding first, carried hidden second (matches the C++ NPU MTP base,
  ``mtp_model_base.h``).
- ``input_embedding`` is the hidden state produced by the target model (or
  the previous draft step) and supplied by the caller; when None the draft
  runs in prefill mode off its own token embedding.
- The draft owns its ``embed_tokens`` / ``lm_head`` copies (the w8a8
  checkpoint materializes them inside the appended layer; the exporter copies
  them for bf16), so no target->draft weight sharing is required.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.qlinear import QLinearWeightLoader
from xllm.python.models.base import PyModelBase
from xllm.python.models.glm5_next import (
    Glm5NextConfig,
    Glm5NextForCausalLM,
    Glm5NextMlaAttention,
    Glm5NextMoE,
    _RMSNorm,
)


class Glm5NextMtpDecoderLayer(nn.Module):
    """One pre-norm decoder layer without mHC (the MTP checkpoint layer shape)."""

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.layer_id = 0
        self.input_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                        dtype, device)
        # Draft schedule: layer 0 is deepseek_sparse_attention (the appended
        # MTP layer carries a full indexer), MoE MLP (first_k_dense_replace=0).
        self.self_attn = Glm5NextMlaAttention(cfg, 0, dtype, device)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size,
                                                 cfg.rms_norm_eps, dtype,
                                                 device)
        self.mlp = Glm5NextMoE(cfg, dtype, device)

    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                prev_topk_indices: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_ids,
                                  attention_mask, prev_topk_indices)
        if isinstance(attn_out, tuple):
            hidden_states, topk = attn_out
        else:
            hidden_states, topk = attn_out, None
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk


class Glm5NextMtpModel(nn.Module):
    """MTP draft body: eh_proj fusion + one decoder layer + shared-head norm."""

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp,
            dtype=dtype, device=device,
        )
        self.enorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Fuses [norm(token_embed) ‖ norm(carried hidden)] -> hidden.
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size, cfg.hidden_size // tp, tp,
            gather_output=True, dtype=dtype, device=device,
        )
        self.layers = nn.ModuleList(
            [Glm5NextMtpDecoderLayer(cfg, dtype, device)
             for _ in range(cfg.n_layers)]
        )
        # Checkpoint name: model.layers.<n>.shared_head.norm — exported as
        # model.norm.weight by the MTP exporter.
        self.norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)

    def forward(self, input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                input_embedding: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # Normalize flat [N] runner inputs to [B, S] like Glm5NextModel.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        token_hidden = self.embed_tokens(input_ids)

        if input_embedding is None:
            carried = token_hidden
        else:
            if input_embedding.dim() == 2:
                input_embedding = input_embedding.view(token_hidden.shape)
            carried = input_embedding

        e = self.enorm(token_hidden)
        h = self.hnorm(carried)
        hidden_states = self.eh_proj(torch.cat((e, h), dim=-1))

        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=hidden_states.device,
        )
        prev_topk = None
        for layer in self.layers:
            hidden_states, prev_topk = layer(
                hidden_states, position_ids, attention_mask, prev_topk
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states.view(-1, self.cfg.hidden_size)


class Glm5NextMtpForCausalLM(Glm5NextForCausalLM):
    """GLM-5-Next MTP draft calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        # Inherit Glm5NextForCausalLM for its weight-loader helpers
        # (_load_dsa_attn / _load_mlp / ...), but deliberately skip its
        # __init__ (it builds the full 45-layer text model). Mirror the VL
        # composer: initialize nn.Module directly.
        nn.Module.__init__(self)
        self.cfg = Glm5NextConfig.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get("tp_rank", 0))
        dtype = self.resolve_dtype(
            config.get("dtype") or config.get("torch_dtype")
        )
        device = torch.device(config.get("device", "cpu"))
        self.dtype = dtype
        self.device = device

        self.model = Glm5NextMtpModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size, self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size, gather_output=True, dtype=dtype, device=device,
        )
        self.to(device=device, dtype=dtype)

    def load_weights(self, state_dicts: list, tp_rank: int,
                     tp_size: int) -> None:
        """Load the exported MTP draft checkpoint (loader-native key names).

        The exporter (tools/export_mtp_glm5_next.py) remaps the appended
        checkpoint layer ``model.language_model.layers.<n>`` to
        ``model.layers.0`` plus top-level ``model.{enorm,hnorm,eh_proj}``,
        ``model.norm`` (shared_head.norm), ``model.embed_tokens`` and
        ``lm_head``, so no real-checkpoint aliasing is needed here.
        """
        L = QLinearWeightLoader(self, state_dicts, tp_size, tp_rank)
        # embed_tokens: HiddenParallelEmbedding — shard the hidden dim.
        L.load_fp("model.embed_tokens.weight", dim=1)
        p = "model.layers.0."
        L.load_fp(p + "input_layernorm.weight")
        L.load_fp(p + "post_attention_layernorm.weight")
        self._load_dsa_attn(L, p + "self_attn.", 0)
        self._load_mlp(L, p + "mlp.", 0)
        L.load_fp("model.norm.weight")
        # lm_head: ColumnParallelLinear — shard the vocab dim.
        L.load_fp("lm_head.weight", dim=0)
        # MTP-specific: replicated norms + column-parallel fusion projection.
        L.load_fp("model.enorm.weight")
        L.load_fp("model.hnorm.weight")
        L.load_fp("model.eh_proj.weight", dim=0)


# Registration is centralised in xllm.python.registry; import there.
