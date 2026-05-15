/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

#pragma once

#include <atb/atb_infer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/interruption_bus.h"
#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_eagle3_decoder_layer_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_pos_embedding_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

namespace {

torch::Tensor build_attention_mask(const ModelInputParams& input_params,
                                   const torch::Tensor& reference,
                                   layer::AttentionMask& attn_mask) {
  if (input_params.batch_forward_type.is_decode()) {
    return torch::Tensor();
  }
  if (FLAGS_enable_chunked_prefill) {
    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(input_params.num_sequences);
    for (int32_t i = 0; i < input_params.num_sequences; ++i) {
      req_mask_vec.emplace_back(
          attn_mask.gen_append_mask(input_params.q_seq_lens_vec[i],
                                    input_params.kv_seq_lens_vec[i],
                                    input_params.kv_max_seq_len,
                                    reference.dtype().toScalarType(),
                                    reference.device()));
    }
    if (!req_mask_vec.empty()) {
      return torch::cat(req_mask_vec, 0);
    }
    return torch::Tensor();
  }
  return attn_mask.get_attn_mask(
      128, reference.dtype().toScalarType(), reference.device());
}

bool has_weight(const StateDict& state_dict) {
  return state_dict.has("weight") || state_dict.has("qweight");
}

void log_kimi_eagle3_tensor(const std::string& name,
                            const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    LOG(INFO) << name << ": undefined";
    return;
  }
  LOG(INFO) << name << ": shape=" << tensor.sizes()
            << ", dtype=" << tensor.scalar_type()
            << ", device=" << tensor.device()
            << ", contiguous=" << tensor.is_contiguous();
}

std::string format_int_vector(const std::vector<int32_t>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

}  // namespace

class KimiK25Eagle3DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit KimiK25Eagle3DecoderLayerImpl(const ModelContext& context,
                                         const int32_t layer_id = 0)
      : layer_id_(layer_id) {
    CHECK(layer_id_ >= 0) << "layer_id must be >= 0, but got " << layer_id_;
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuEagle3DecoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& hidden_states_extra,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event,
                        std::atomic<bool>* event_flag) {
    return decoder_layer_(hidden_states,
                          hidden_states_extra,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag,
                          layer_id_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights();
    UNUSED_PARAMETER(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }

  void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(state_dict);
  }

 private:
  layer::NpuEagle3DecoderLayer decoder_layer_{nullptr};
  int32_t layer_id_;
};
TORCH_MODULE(KimiK25Eagle3DecoderLayer);

class KimiK25Eagle3ModelImpl : public torch::nn::Module {
 public:
  explicit KimiK25Eagle3ModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    const ParallelArgs& parallel_args = context.get_parallel_args();
    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;

    embed_tokens_ =
        register_module("embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args_.head_dim(),
        model_args_.max_position_embeddings(),
        model_args_.rope_theta(),
        options_);

    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options_.device(),
                                      options_.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    fc_ = register_module("fc", layer::NpuColumnParallelLinear(context));
    decoder_ = register_module("midlayer", KimiK25Eagle3DecoderLayer(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, 0);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);

    if (dp_size_ > 1 && tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }

    torch::Tensor hidden_states = embed_tokens_(tokens, 0);
    torch::Tensor hidden_states_extra = input_params.input_embedding;
    LOG(INFO) << "KimiK25Eagle3 forward:"
              << " phase="
              << (input_params.batch_forward_type.is_decode() ? "decode"
                                                              : "prefill")
              << ", num_sequences=" << input_params.num_sequences
              << ", kv_max_seq_len=" << input_params.kv_max_seq_len
              << ", q_seq_lens_vec_size=" << input_params.q_seq_lens_vec.size()
              << ", kv_seq_lens_vec_size="
              << input_params.kv_seq_lens_vec.size();
    log_kimi_eagle3_tensor("  tokens", tokens);
    log_kimi_eagle3_tensor("  positions", positions);
    log_kimi_eagle3_tensor("  hidden_states", hidden_states);
    log_kimi_eagle3_tensor("  input_embedding", hidden_states_extra);
    if (!hidden_states_extra.defined() || hidden_states_extra.numel() == 0) {
      LOG(WARNING) << "KimiK25Eagle3 hidden_states_extra is missing, "
                   << "falling back to hidden_states.";
      hidden_states_extra = hidden_states;
    }
    LOG(INFO) << "KimiK25Eagle3 hidden_states_extra rows check:"
              << " hidden_states_rows=" << hidden_states.size(0)
              << ", hidden_states_extra_rows=" << hidden_states_extra.size(0)
              << ", num_sequences=" << input_params.num_sequences
              << ", q_seq_lens_vec="
              << format_int_vector(input_params.q_seq_lens_vec);
    CHECK_EQ(hidden_states_extra.size(0), hidden_states.size(0))
        << "KimiK25Eagle3 hidden_states_extra row count mismatch, "
        << "hidden_states=" << hidden_states.sizes()
        << ", hidden_states_extra=" << hidden_states_extra.sizes();
    log_kimi_eagle3_tensor("  hidden_states_extra_checked",
                           hidden_states_extra);
    if (hidden_states_extra.size(-1) != hidden_states.size(-1)) {
      LOG(INFO) << "KimiK25Eagle3 applying fc to hidden_states_extra:"
                << " input_dim=" << hidden_states_extra.size(-1)
                << ", target_dim=" << hidden_states.size(-1);
      hidden_states_extra = fc_(hidden_states_extra, 0);
      log_kimi_eagle3_tensor("  hidden_states_extra_after_fc",
                             hidden_states_extra);
    }

    torch::Tensor target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    torch::Tensor attn_mask;
    if (!input_params.batch_forward_type.is_decode()) {
      attn_mask = build_attention_mask(input_params, cos_pos, attn_mask_);
    }

    aclrtEvent* event{nullptr};
    std::atomic<bool>* event_flag{nullptr};
    if (input_params.layer_synchronizer != nullptr) {
      event = input_params.layer_synchronizer->get_event(0);
      event_flag = input_params.layer_synchronizer->get_event_flag(0);
    }
    if (!input_params.synchronize_layer(0)) {
      return ModelOutput();
    }

    CHECK_EQ(kv_caches.size(), 1U);
    decoder_(hidden_states,
             hidden_states_extra,
             cos_pos,
             sin_pos,
             attn_mask,
             kv_caches[0],
             input_params_new,
             event,
             event_flag);
    torch::Tensor aux_hidden_states = hidden_states.clone();
    hidden_states = norm_(hidden_states, 0);

    return ModelOutput(hidden_states,
                       /*residual=*/torch::Tensor(),
                       /*aux_hidden_states=*/aux_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    StateDict embed_dict = state_dict.get_dict_with_prefix("embed_tokens.");
    if (has_weight(embed_dict)) {
      embed_tokens_->load_state_dict(embed_dict);
      embed_tokens_loaded_ = true;
    }

    StateDict fc_dict = state_dict.get_dict_with_prefix("fc.");
    if (has_weight(fc_dict)) {
      fc_->load_state_dict(fc_dict);
    }

    decoder_->load_state_dict(state_dict.get_dict_with_prefix("midlayer."));

    StateDict norm_dict = state_dict.get_dict_with_prefix("norm.");
    if (has_weight(norm_dict)) {
      norm_->load_state_dict(norm_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    fc_->verify_loaded_weights(prefix + "fc.");
    decoder_->verify_loaded_weights(prefix + "midlayer.");
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    embed_tokens_->merge_loaded_weights();
    fc_->merge_loaded_weights();
    decoder_->merge_loaded_weights();
    norm_->merge_loaded_weights();
  }

  layer::NpuWordEmbedding get_npu_word_embedding() { return embed_tokens_; }

  void set_npu_word_embedding(layer::NpuWordEmbedding& word_embedding) {
    if (!embed_tokens_loaded_) {
      embed_tokens_ = word_embedding;
    }
  }

 private:
  ModelArgs model_args_;
  int32_t dp_rank_ = 0;
  int32_t dp_size_ = 1;
  int32_t dp_local_tp_size_ = 1;
  torch::TensorOptions options_;
  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::NpuWordEmbedding embed_tokens_{nullptr};
  layer::NpuColumnParallelLinear fc_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};
  KimiK25Eagle3DecoderLayer decoder_{nullptr};
  bool embed_tokens_loaded_ = false;
};
TORCH_MODULE(KimiK25Eagle3Model);

class KimiK25Eagle3ForCausalLMImpl : public torch::nn::Module {
 public:
  explicit KimiK25Eagle3ForCausalLMImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    tie_word_embeddings_ = model_args.tie_word_embeddings();

    model_ = register_module("model", KimiK25Eagle3Model(context));
    npu_lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));

    load_lm_head_from_target_ = false;
    if (!tie_word_embeddings_) {
      int64_t vocab_size = model_args.vocab_size();
      if (vocab_size == 0) {
        load_lm_head_from_target_ = true;
      }
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return npu_lm_head_(hidden_states, seleted_idxes, 0);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto sub_dict = state_dict->get_dict_with_prefix(prefix + "model.");
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix(prefix);
      }
      model_->load_state_dict(sub_dict);

      if (!load_lm_head_from_target_) {
        if (tie_word_embeddings_) {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
        } else {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix("lm_head."));
        }
      }
    }

    model_->verify_loaded_weights(prefix);
    if (!load_lm_head_from_target_) {
      if (tie_word_embeddings_) {
        npu_lm_head_->verify_loaded_weights(prefix + "embed_tokens.");
      } else {
        npu_lm_head_->verify_loaded_weights("lm_head.");
      }
    }
    model_->merge_loaded_weights();
    if (!load_lm_head_from_target_) {
      npu_lm_head_->merge_loaded_weights();
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    UNUSED_PARAMETER(layer_id);
    UNUSED_PARAMETER(expert_ids);
  }

  void update_expert_weight(int32_t layer_id) { UNUSED_PARAMETER(layer_id); }

  layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    if (load_lm_head_from_target_) {
      npu_lm_head_ = head;
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  KimiK25Eagle3Model model_{nullptr};
  int device_id_ = 0;
  bool tie_word_embeddings_{false};
  bool load_lm_head_from_target_{false};
  layer::NpuLmHead npu_lm_head_{nullptr};
};
TORCH_MODULE(KimiK25Eagle3ForCausalLM);

namespace {

bool load_kimi_k25_eagle3_tokenizer_args(const JsonReader& json,
                                         TokenizerArgs* args) {
  auto parent_loader = ModelRegistry::get_tokenizer_args_loader("kimi_k25");
  CHECK(parent_loader != nullptr)
      << "Tokenizer args loader for kimi_k25 must be registered first";
  return parent_loader(json, args);
}

}  // namespace

REGISTER_CAUSAL_MODEL(kimi_k25_eagle3, KimiK25Eagle3ForCausalLM);

REGISTER_TOKENIZER_ARGS_LOADER(kimi_k25_eagle3,
                               load_kimi_k25_eagle3_tokenizer_args);

REGISTER_MODEL_ARGS(kimi_k25_eagle3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k25_eagle3");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });
  LOAD_ARG_OR(vocab_size, "vocab_size", 163840);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06f);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(draft_vocab_size, "draft_vocab_size", 0);
  if (args->draft_vocab_size() > 0) {
    args->vocab_size(args->draft_vocab_size());
  }
  LOAD_ARG_OR_FUNC(layers_to_capture, "layers_to_capture", [&] {
    if (auto layer_ids = json.value<std::vector<int32_t>>("layers_to_capture");
        layer_ids.has_value()) {
      return layer_ids.value();
    }
    if (auto layer_ids =
            json.value<std::vector<int32_t>>("text_config.layers_to_capture");
        layer_ids.has_value()) {
      return layer_ids.value();
    }
    if (auto layer_ids = json.value<std::vector<int32_t>>(
            "eagle_aux_hidden_state_layer_ids");
        layer_ids.has_value()) {
      return layer_ids.value();
    }
    return std::vector<int32_t>{};
  });
  if (!args->layers_to_capture().empty()) {
    LOG(INFO) << "KimiK25Eagle3 layers_to_capture from config: "
              << format_int_vector(args->layers_to_capture());
  }
  LOAD_ARG_OR_FUNC(bos_token_id, "bos_token_id", [&] {
    return json.value_or<int32_t>("text_config.bos_token_id", 163584);
  });
  LOAD_ARG_OR_FUNC(eos_token_id, "eos_token_id", [&] {
    return json.value_or<int32_t>("text_config.eos_token_id", 163585);
  });
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model
