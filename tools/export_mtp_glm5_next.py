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
"""Export the GLM-5-Next MTP draft layer into a standalone draft-model dir.

The appended draft layer lives in the main checkpoint as
``model.language_model.layers.<num_hidden_layers>`` (a full DSA sparse-
attention MoE layer with indexer, WITHOUT mHC residual streams) plus its
``enorm`` / ``hnorm`` / ``eh_proj`` fusion weights and ``shared_head.norm``.
The draft engine loads this dir with ``model_type=glm5_next_mtp`` and the
python draft model (``xllm/python/models/glm5_next_mtp.py``).

Output layout (loader-native names, no aliasing needed at load time):
  - ``model.layers.0.*``                <- ``model.language_memory.layers.<n>.*``
  - ``model.{enorm,hnorm,eh_proj}.weight`` <- layer-local MTP fusion weights
  - ``model.norm.weight``               <- ``...layers.<n>.shared_head.norm.weight``
  - ``model.embed_tokens.weight``       <- w8a8: layer-local copy; bf16: main embed
  - ``lm_head.weight``                  <- w8a8: ``...shared_head.head.weight``;
                                           bf16: main ``lm_head.weight``
Quantized (w8a8) expert weights and their scales are copied as-is — the
python loader probes them per module.

Usage:
  python3 tools/export_mtp_glm5_next.py \
      --input-dir /export/home/models/GLM-next-w8a8 \
      --output-dir /export/home/models/GLM-next-w8a8-mtp
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger  # noqa: E402

DRAFT_MODEL_TYPE = "glm5_next_mtp"
DRAFT_ARCHITECTURE = "Glm5NextMtpForCausalLM"
TEXT_PREFIX = "model.language_model."
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)


def _load_index(model_dir: str) -> dict:
    """key -> shard file map for the checkpoint (works for both dirs)."""
    for name in ("quant_model_weights.safetensors.index.json",
                 "model.safetensors.index.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                index = json.load(f)
            return dict(index["weight_map"])
    raise FileNotFoundError(f"no safetensors index found in {model_dir}")


def _iter_layer_keys(weight_map: dict, layer_idx: int):
    prefix = f"{TEXT_PREFIX}layers.{layer_idx}."
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            yield key, shard


def export_draft(input_dir: str, output_dir: str) -> None:
    from safetensors.torch import load_file, save_file

    with open(os.path.join(input_dir, "config.json")) as f:
        config = json.load(f)
    text_config = dict(config.get("text_config", config))
    n_layers = int(text_config["num_hidden_layers"])
    mtp_layer = n_layers  # appended MTP layer index

    weight_map = _load_index(input_dir)

    # Layer-scope keys promoted to top level by the remap below (enorm/hnorm/
    # eh_proj/shared_head/embed_tokens) — excluded here to avoid duplicates.
    promoted = ("enorm.", "hnorm.", "eh_proj.", "shared_head.", "embed_tokens.")

    tensors: dict = {}
    for key, shard in _iter_layer_keys(weight_map, mtp_layer):
        local = key[len(f"{TEXT_PREFIX}layers.{mtp_layer}."):]
        if local.startswith(promoted):
            continue
        tensors["model.layers.0." + local] = key

    # Shared/embedding tensors resolve in priority order: layer-local copies
    # first (the w8a8 pipeline materializes them), then the main checkpoint.
    def _resolve(candidates: list, name: str) -> str:
        for cand in candidates:
            if cand in weight_map:
                return cand
        raise KeyError(f"missing required MTP weight: {name}")

    remap = {
        "model.enorm.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.enorm.weight"],
        "model.hnorm.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.hnorm.weight"],
        "model.eh_proj.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.eh_proj.weight"],
        "model.norm.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.shared_head.norm.weight",
             f"{TEXT_PREFIX}norm.weight"],
        "model.embed_tokens.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.embed_tokens.weight",
             f"{TEXT_PREFIX}embed_tokens.weight"],
        "lm_head.weight":
            [f"{TEXT_PREFIX}layers.{mtp_layer}.shared_head.head.weight",
             "lm_head.weight"],
    }
    for new_key, candidates in remap.items():
        tensors[new_key] = _resolve(candidates, new_key)

    os.makedirs(output_dir, exist_ok=True)
    shard_cache: dict = {}
    out_tensors: dict = {}
    for new_key, src_key in tensors.items():
        shard = weight_map[src_key]
        if shard not in shard_cache:
            shard_cache[shard] = load_file(os.path.join(input_dir, shard))
            logger.info(f"loaded shard {shard}")
        out_tensors[new_key] = shard_cache[shard][src_key]
    save_file(out_tensors, os.path.join(output_dir, "model.safetensors"))
    with open(os.path.join(output_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump({"metadata": {}, "weight_map":
                   {k: "model.safetensors" for k in out_tensors}}, f)

    # Draft config: one DSA layer, no dense prefix, no mHC, no nested MTP.
    draft_text = dict(text_config)
    draft_text["num_hidden_layers"] = 1
    draft_text["num_nextn_predict_layers"] = 0
    draft_text["layer_types"] = ["deepseek_sparse_attention"]
    draft_text["mlp_layer_types"] = ["sparse"]
    draft_text["indexer_types"] = ["full"]
    draft_text["first_k_dense_replace"] = 0
    draft_text["model_type"] = "glm5_next_text"
    draft_config = dict(config)
    draft_config["model_type"] = DRAFT_MODEL_TYPE
    draft_config["architectures"] = [DRAFT_ARCHITECTURE]
    draft_config["text_config"] = draft_text
    draft_config.pop("vision_config", None)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(draft_config, f, indent=2, ensure_ascii=False)

    for name in TOKENIZER_FILES:
        src = os.path.join(input_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, name))

    n_quant = sum(1 for k in out_tensors if k.endswith("weight_scale"))
    logger.info(f"exported {len(out_tensors)} tensors "
                f"({n_quant} quant scales) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                        help="GLM-5-Next checkpoint dir (bf16 or w8a8)")
    parser.add_argument("--output-dir", required=True,
                        help="draft model output dir")
    args = parser.parse_args()
    export_draft(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
