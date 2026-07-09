/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <sstream>

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/metrics.h"
#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/util/utils.h"
#include "platform/npu/device_capture_lock.h"

namespace xllm::npu {

namespace {
constexpr uint64_t kSpecVerifyGraphKeyMask = 1ull << 63;
constexpr uint64_t kSpecVerifyQMaxSeqLenShift = 32;

std::pair<torch::Tensor, torch::Tensor> find_attention_plan_kv_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return {std::move(k_cache), std::move(v_cache)};
    }
  }
  return {torch::Tensor(), torch::Tensor()};
}

void append_tensor_key(std::ostringstream& os,
                       const char* name,
                       const torch::Tensor& tensor) {
  os << '|' << name << ':';
  if (!tensor.defined()) {
    os << "undef";
    return;
  }
  os << static_cast<int>(tensor.scalar_type()) << ':';
  for (const auto dim : tensor.sizes()) {
    os << dim << ',';
  }
}

void append_int_vec_key(std::ostringstream& os,
                        const char* name,
                        const std::vector<int32_t>& values) {
  os << '|' << name << ':';
  for (const auto value : values) {
    os << value << ',';
  }
}

void append_tensor_vec_key(std::ostringstream& os,
                           const char* name,
                           const std::vector<torch::Tensor>& tensors) {
  os << '|' << name << "_size:" << tensors.size();
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::string item_name = std::string(name) + "_" + std::to_string(i);
    append_tensor_key(os, item_name.c_str(), tensors[i]);
  }
}

int64_t get_onerec_decoder_output_tokens(const torch::Tensor& tokens,
                                         const ModelInputParams& params,
                                         const ModelArgs& args) {
  const auto* onerec_params = params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->decoder_context_embedding.defined()) {
    const int64_t hidden_size = std::max<int64_t>(1, args.hidden_size());
    return onerec_params->decoder_context_embedding.numel() / hidden_size;
  }
  return tokens.defined() && tokens.dim() >= 1 ? tokens.size(0) : 0;
}

std::string build_onerec_prefill_graph_key(const torch::Tensor& tokens,
                                           const torch::Tensor& positions,
                                           const ModelInputParams& params,
                                           const ModelArgs& args) {
  const auto* onerec_params = params.onerec_params();
  CHECK(onerec_params != nullptr);
  std::ostringstream os;
  os << "onerec_prefill"
     << "|out_tokens:" << get_onerec_decoder_output_tokens(tokens, params, args)
     << "|is_first:" << onerec_params->is_first_prefill
     << "|has_encoder_output:" << onerec_params->has_encoder_output
     << "|bs:" << onerec_params->bs
     << "|group_width:" << onerec_params->group_width
     << "|seq_len:" << onerec_params->seq_len
     << "|encoder_max_seq_len:" << onerec_params->encoder_max_seq_len
     << "|num_sequences:" << params.meta.num_sequences
     << "|q_max_seq_len:" << params.meta.q_max_seq_len
     << "|kv_max_seq_len:" << params.meta.kv_max_seq_len
     << "|use_xattn:" << (params.onerec_xattention_params() != nullptr)
     << "|use_moe:" << args.use_moe();
  append_tensor_key(os, "tokens", tokens);
  append_tensor_key(os, "positions", positions);
  append_tensor_key(
      os, "decoder_context", onerec_params->decoder_context_embedding);
  append_tensor_key(
      os, "encoder_seq_lens_tensor", onerec_params->encoder_seq_lens_tensor);
  append_tensor_key(
      os, "cross_kv_cu", onerec_params->cross_attn_kv_cu_seq_lens);
  append_tensor_key(
      os, "cross_new_slots", onerec_params->cross_attn_new_cache_slots);
  append_tensor_key(
      os, "cross_block_tables", onerec_params->cross_attn_block_tables);
  append_int_vec_key(os, "encoder_seq_lens", onerec_params->encoder_seq_lens);
  append_int_vec_key(
      os, "cross_kv_cu_vec", onerec_params->cross_attn_kv_cu_seq_lens_vec);

  if (const auto* xattn_params = params.onerec_xattention_params()) {
    append_tensor_vec_key(os, "unshared_k", xattn_params->unshared_k_caches);
    append_tensor_vec_key(os, "unshared_v", xattn_params->unshared_v_caches);
    append_tensor_vec_key(os, "shared_k", xattn_params->shared_k_caches);
    append_tensor_vec_key(os, "shared_v", xattn_params->shared_v_caches);
    append_tensor_key(os, "beam_width", xattn_params->beam_width_tensor);
    append_tensor_key(os, "current_round", xattn_params->current_round_tensor);
  }
  return os.str();
}

bool is_onerec_decoder_prefill_graph_candidate(
    CausalLM* model,
    const ModelInputParams& params,
    bool enable_onerec_prefill_acl_graph) {
  const auto* onerec_params = params.onerec_params();
  if (!enable_onerec_prefill_acl_graph || onerec_params == nullptr ||
      !model->supports_onerec_prefill_graph()) {
    return false;
  }
  if (onerec_params->is_encoder_forward ||
      onerec_params->rec_stage != OneRecModelInputParams::RecStage::PREFILL) {
    return false;
  }
  return params.parallel.layer_synchronizer == nullptr;
}

}  // namespace

class OneRecPrefillGraphParam {
 public:
  OneRecPrefillGraphParam(const ModelArgs& args,
                          const torch::Device& device,
                          const runtime::Options& options)
      : args_(args), device_(device), options_(options) {}

  ModelInputParams update(CausalLM* model,
                          const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          const ModelInputParams& params,
                          const torch::Tensor& encoder_output) {
    copy_tensor(tokens, persistent_tokens_);
    copy_tensor(positions, persistent_positions_);
    copy_tensor(encoder_output, persistent_encoder_output_);

    params_for_capture_ = params;
    auto& onerec_params = params_for_capture_.mutable_onerec_params();
    copy_tensor(onerec_params.decoder_context_embedding,
                persistent_decoder_context_embedding_);
    copy_tensor(onerec_params.encoder_seq_lens_tensor,
                persistent_encoder_seq_lens_tensor_);
    copy_tensor(onerec_params.cross_attn_kv_cu_seq_lens,
                persistent_cross_attn_kv_cu_seq_lens_);
    copy_tensor(onerec_params.cross_attn_new_cache_slots,
                persistent_cross_attn_new_cache_slots_);
    copy_tensor(onerec_params.cross_attn_block_tables,
                persistent_cross_attn_block_tables_);

    if (persistent_decoder_context_embedding_.defined()) {
      onerec_params.decoder_context_embedding =
          persistent_decoder_context_embedding_;
    }
    if (persistent_encoder_seq_lens_tensor_.defined()) {
      onerec_params.encoder_seq_lens_tensor =
          persistent_encoder_seq_lens_tensor_;
    }
    if (persistent_cross_attn_kv_cu_seq_lens_.defined()) {
      onerec_params.cross_attn_kv_cu_seq_lens =
          persistent_cross_attn_kv_cu_seq_lens_;
    }
    if (persistent_cross_attn_new_cache_slots_.defined()) {
      onerec_params.cross_attn_new_cache_slots =
          persistent_cross_attn_new_cache_slots_;
    }
    if (persistent_cross_attn_block_tables_.defined()) {
      onerec_params.cross_attn_block_tables =
          persistent_cross_attn_block_tables_;
    }

    ensure_hidden_states(tokens, params);
    ensure_cross_kv_caches(params, encoder_output);
    bind_model(model, onerec_params.is_first_prefill);
    return params_for_capture_;
  }

  torch::Tensor tokens() const { return persistent_tokens_; }
  torch::Tensor positions() const { return persistent_positions_; }
  torch::Tensor hidden_states() const { return hidden_states_; }

  void set_hidden_states(const torch::Tensor& value) {
    CHECK(hidden_states_.defined());
    CHECK_EQ(hidden_states_.sizes(), value.sizes());
    hidden_states_.copy_(value, /*non_blocking=*/true);
  }

 private:
  static void copy_tensor(const torch::Tensor& src, torch::Tensor& dst) {
    if (!src.defined()) {
      dst = torch::Tensor();
      return;
    }
    if (!dst.defined()) {
      dst = torch::empty_like(src);
    }
    CHECK_EQ(dst.sizes(), src.sizes());
    CHECK_EQ(dst.scalar_type(), src.scalar_type());
    CHECK_EQ(dst.device(), src.device());
    if (src.numel() > 0) {
      dst.copy_(src, /*non_blocking=*/true);
    }
  }

  void ensure_hidden_states(const torch::Tensor& tokens,
                            const ModelInputParams& params) {
    const int64_t output_tokens =
        get_onerec_decoder_output_tokens(tokens, params, args_);
    CHECK_GT(output_tokens, 0) << "OneRec prefill graph output is empty.";
    const auto options = torch::TensorOptions()
                             .dtype(util::parse_dtype(args_.dtype(), device_))
                             .device(device_);
    const std::vector<int64_t> shape = {output_tokens, args_.hidden_size()};
    if (!hidden_states_.defined() || hidden_states_.sizes().vec() != shape) {
      hidden_states_ = torch::empty(shape, options);
    }
  }

  void ensure_cross_kv_caches(const ModelInputParams& params,
                              const torch::Tensor& encoder_output) {
    const auto* onerec_params = params.onerec_params();
    if (onerec_params == nullptr || !onerec_params->is_first_prefill ||
        !encoder_output.defined()) {
      return;
    }
    const int64_t bs = encoder_output.size(0);
    const int64_t seq_len = encoder_output.size(1);
    const auto decoder_kv_heads = args_.decoder_n_kv_heads().has_value()
                                      ? args_.decoder_n_kv_heads()
                                      : args_.n_kv_heads();
    const int64_t kv_heads = decoder_kv_heads.value_or(args_.decoder_n_heads());
    const int64_t kv_heads_per_rank =
        kv_heads / std::max<int32_t>(1, options_.tp_size());
    const int64_t kv_hidden_size = kv_heads_per_rank * args_.decoder_head_dim();
    const std::vector<int64_t> shape = {bs, seq_len, kv_hidden_size};
    const auto options = torch::TensorOptions()
                             .dtype(encoder_output.dtype())
                             .device(encoder_output.device());
    const size_t layer_num = static_cast<size_t>(args_.n_layers());
    cross_k_caches_.resize(layer_num);
    cross_v_caches_.resize(layer_num);
    for (size_t i = 0; i < layer_num; ++i) {
      if (!cross_k_caches_[i].defined() ||
          cross_k_caches_[i].sizes().vec() != shape ||
          cross_k_caches_[i].dtype() != encoder_output.dtype() ||
          cross_k_caches_[i].device() != encoder_output.device()) {
        cross_k_caches_[i] = torch::empty(shape, options);
      }
      if (!cross_v_caches_[i].defined() ||
          cross_v_caches_[i].sizes().vec() != shape ||
          cross_v_caches_[i].dtype() != encoder_output.dtype() ||
          cross_v_caches_[i].device() != encoder_output.device()) {
        cross_v_caches_[i] = torch::empty(shape, options);
      }
    }
  }

  void bind_model(CausalLM* model, bool bind_cross_kv_caches) {
    model->bind_onerec_prefill_graph_buffers(
        persistent_encoder_output_,
        bind_cross_kv_caches ? cross_k_caches_ : empty_tensor_vec_,
        bind_cross_kv_caches ? cross_v_caches_ : empty_tensor_vec_);
  }

  const ModelArgs& args_;
  torch::Device device_;
  runtime::Options options_;

  ModelInputParams params_for_capture_;
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_encoder_output_;
  torch::Tensor persistent_decoder_context_embedding_;
  torch::Tensor persistent_encoder_seq_lens_tensor_;
  torch::Tensor persistent_cross_attn_kv_cu_seq_lens_;
  torch::Tensor persistent_cross_attn_new_cache_slots_;
  torch::Tensor persistent_cross_attn_block_tables_;
  std::vector<torch::Tensor> cross_k_caches_;
  std::vector<torch::Tensor> cross_v_caches_;
  std::vector<torch::Tensor> empty_tensor_vec_;
  torch::Tensor hidden_states_;
};

class OneRecPrefillAclGraph {
 public:
  OneRecPrefillAclGraph(const ModelArgs& args,
                        const torch::Device& device,
                        const runtime::Options& options)
      : param_(args, device, options), device_index_(device.index()) {
    capture_stream_ = c10_npu::getStreamFromPool(true, device_index_);
  }

  bool capture(CausalLM* model,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               std::vector<KVCache>& kv_caches,
               const ModelInputParams& params,
               const torch::Tensor& encoder_output) {
    torch::npu::synchronize();
    auto params_for_capture =
        param_.update(model, tokens, positions, params, encoder_output);
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_index_).stream();
    aclrtSynchronizeStream(stream);

    bool need_restore_stream = false;
    {
      auto& capture_lock =
          ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
              device_index_);
      std::lock_guard<std::mutex> lock_guard(capture_lock);
      if (c10_npu::getCurrentNPUStream(device_index_) ==
          c10_npu::getDefaultNPUStream(device_index_)) {
        c10_npu::setCurrentNPUStream(capture_stream_.value());
        aclrtSynchronizeStream(capture_stream_.value().stream());
        need_restore_stream = true;
      }

      graph_.capture_begin(
          {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
      auto forward_result = model->forward(
          param_.tokens(), param_.positions(), kv_caches, params_for_capture);
      param_.set_hidden_states(forward_result.hidden_states);
      graph_.capture_end();

      if (need_restore_stream) {
        c10_npu::setCurrentNPUStream(
            c10_npu::getDefaultNPUStream(device_index_));
      }
    }
    aclrtSynchronizeStream(stream);
    graph_.replay();
    captured_ = true;
    return true;
  }

  ModelOutput replay(CausalLM* model,
                     const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     const ModelInputParams& params,
                     const torch::Tensor& encoder_output) {
    CHECK(captured_);
    param_.update(model, tokens, positions, params, encoder_output);
    graph_.replay();
    return ModelOutput(param_.hidden_states());
  }

  ModelOutput output() const { return ModelOutput(param_.hidden_states()); }

 private:
  OneRecPrefillGraphParam param_;
  c10_npu::NPUGraph graph_;
  std::optional<c10_npu::NPUStream> capture_stream_;
  c10::DeviceIndex device_index_;
  bool captured_ = false;
};

bool AclGraph::capture(CausalLM* model,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens) {
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // For hybrid models (e.g., qwen3_next with mixed GDN/full_attention layers),
  // we need to find the first Full Attention layer to get the correct kv_cache.
  // GDN layers have empty key_cache_/value_cache_ while Full Attention layers
  // have valid kv caches. Using layer 0's cache directly would be incorrect
  // if layer 0 is a GDN layer.
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";
  prepare_model_graph_metadata(
      model,
      persistent_param_.persistent_positions(num_tokens_),
      graph_params.value());

  if (model->is_hybrid_linear_attention()) {
    graph_task_context_ = std::make_shared<AclGraphTaskUpdateContext>();
    graph_task_context_->begin_capture();
    graph_params->graph.acl_graph_task_update_context = graph_task_context_;
  }

  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Acquire device-level lock to prevent prepare_work_before_execute from
  // executing simultaneously, which would trigger synchronous operations
  // that conflict with capture mode
  auto device_idx = tensor_options.device().index();
  Device::empty_cache(device_idx);

  bool need_restore_stream = false;
  graph_stream_ = stream;

  // capture lock scope
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_idx);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    if (c10_npu::getCurrentNPUStream(device_idx) ==
        c10_npu::getDefaultNPUStream(device_idx)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      graph_stream_ = capture_stream_.value().stream();
      need_restore_stream = true;
    }
    LOG(INFO) << "capture begin, bucket_num_tokens: " << bucket_num_tokens
              << ", actual_num_tokens: " << actual_num_tokens;

    // no mempool id, will create a new one; capture mode is thread local, allow
    // other threads to execute synchronous operations
    graph_.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    // Execute forward pass - NPUGraph mempool manages temporary tensors
    auto forward_result =
        model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                       {persistent_param_.persistent_positions(num_tokens_)},
                       kv_cache,
                       {graph_params.value()});

    // Store result in persistent buffer owned by NPUGraph mempool
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    graph_.capture_end();
    if (graph_task_context_ != nullptr) {
      graph_task_context_->end_capture();
    }
    // Lock is automatically released here when lock goes out of scope
    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(
          c10_npu::getDefaultNPUStream(tensor_options.device().index()));
    }
  }
  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(graph_stream_);
  aclrtSynchronizeStream(stream);
  graph_.replay();
  update_graph_tasks(graph_params.value());
  make_current_stream_wait_for_graph(stream);
  return true;
}

void AclGraph::update_graph_tasks(const ModelInputParams& params) {
  if (graph_task_context_ == nullptr ||
      graph_task_context_->causal_conv1d_tasks.empty()) {
    return;
  }

  const std::vector<int64_t> empty_host_args;
  CHECK(!params.parallel.query_start_loc.empty())
      << "causal_conv1d graph update requires padded query_start_loc";
  CHECK(!params.embedding.linear_state_ids.empty())
      << "causal_conv1d graph update requires padded cache indices";

  std::vector<int64_t> linear_state_indices_host(
      params.embedding.linear_state_ids.begin(),
      params.embedding.linear_state_ids.end());

  c10_npu::NPUStream update_stream = update_stream_.value();
  c10_npu::NPUStreamGuard stream_guard(update_stream);

  for (auto& task : graph_task_context_->causal_conv1d_tasks) {
    CHECK_EQ(params.parallel.query_start_loc.back(), task.x.size(0))
        << "causal_conv1d graph update host args must be padded to the "
           "capture x.shape[0]";
    CHECK_EQ(linear_state_indices_host.size() + 1,
             params.parallel.query_start_loc.size())
        << "cache_indices must be sequence-scoped";

    const std::vector<int64_t>& num_accepted_tokens =
        task.branch == CausalConv1dGraphBranch::kSpecVerify
            ? params.num_accepted_tokens_host
            : empty_host_args;
    if (task.branch == CausalConv1dGraphBranch::kSpecVerify) {
      CHECK_EQ(num_accepted_tokens.size(), linear_state_indices_host.size())
          << "spec causal_conv1d graph update requires accepted-token counts";
    }

    c10_npu::graph_task_update_begin(update_stream, task.handle);
    xllm::kernel::causal_conv1d_out(
        task.output,
        task.x,
        task.weight,
        task.conv_state,
        task.bias,
        torch::IntArrayRef(params.parallel.query_start_loc),
        torch::IntArrayRef(linear_state_indices_host),
        torch::IntArrayRef(empty_host_args),
        torch::IntArrayRef(num_accepted_tokens),
        task.activation_mode,
        task.pad_slot_id,
        task.run_mode);
    c10_npu::graph_task_update_end(update_stream);
    if (task.event != nullptr) {
      task.event->record(update_stream);
    }
  }
}

AclGraph::~AclGraph() {
  if (graph_stream_ != nullptr) {
    aclrtSynchronizeStream(graph_stream_);
  } else if (capture_stream_.has_value()) {
    aclrtSynchronizeStream(capture_stream_.value().stream());
  }
  if (replay_done_event_ != nullptr) {
    aclrtDestroyEvent(replay_done_event_);
    replay_done_event_ = nullptr;
  }
}

void AclGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  // Get a secondary stream from high-priority pool for graph capture.
  // This is required because NPUGraph::capture_begin() enforces that capture
  // must be performed on a non-default stream (see
  // torch_npu/csrc/core/npu/NPUGraph.cpp:159).
  capture_stream_ = c10_npu::getStreamFromPool(true, device_index);
  update_stream_ = c10_npu::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_done_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay completion event";
  LOG(INFO) << "Initialized capture_stream"
            << ", id: " << capture_stream_.value().id()
            << ", device_index: " << device_index;
}

void AclGraph::make_current_stream_wait_for_graph(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_done_event_, nullptr)
      << "replay_done_event is not initialized";
  CHECK_EQ(aclrtRecordEvent(replay_done_event_, graph_stream_), ACL_SUCCESS)
      << "aclrtRecordEvent(replay_done_event) failed";
  if (current_stream != graph_stream_) {
    CHECK_EQ(aclrtStreamWaitEvent(current_stream, replay_done_event_),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(current_stream, replay_done_event) failed";
  }
}

void AclGraph::prepare_model_graph_metadata(CausalLM* model,
                                            const torch::Tensor& positions,
                                            ModelInputParams& params) {
  CHECK(model != nullptr) << "ACL graph model must not be null";
  if (!model->requires_graph_forward_metadata()) {
    return;
  }
  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
    CHECK(model_graph_metadata_state_)
        << "ACL graph metadata state must be initialized during capture";
  }
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(), positions, params);
  CHECK(params.attn_metadata)
      << "model graph metadata preparation did not populate attn_metadata";
}

ModelOutput AclGraph::replay(CausalLM* model,
                             const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // Note: tiling_data is updated in update() if needed - for hybrid models
  // (e.g., qwen3_next with mixed GDN/attention layers), tiling should only
  // be updated when Full Attention layers are involved, which is determined
  // by k_cache being valid and non-empty
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const bool needs_graph_metadata = model->requires_graph_forward_metadata() ||
                                    model->is_hybrid_linear_attention();
  std::optional<ModelInputParams> graph_params =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               num_tokens_,
                               needs_graph_metadata);
  if (needs_graph_metadata) {
    CHECK(graph_params.has_value())
        << "ACL graph replay requires persistent params for graph metadata";
    prepare_model_graph_metadata(
        model,
        persistent_param_.persistent_positions(num_tokens_),
        graph_params.value());
  }

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();
  if (model->is_hybrid_linear_attention()) {
    CHECK(graph_params.has_value())
        << "update() should return ModelInputParams for graph task update";
    update_graph_tasks(graph_params.value());
  }
  make_current_stream_wait_for_graph(stream);

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in AclGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  const bool need_update_attn_mask = model->is_hybrid_linear_attention();
  const bool is_hybrid_linear_attn = model->is_hybrid_linear_attention();
  persistent_param_ = std::make_unique<GraphPersistentParam>(
      args_, device_, options_, need_update_attn_mask, is_hybrid_linear_attn);
}

AclGraphExecutorImpl::~AclGraphExecutorImpl() = default;

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput AclGraphExecutorImpl::run_onerec_prefill_graph(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  const auto* onerec_params = params.onerec_params();
  CHECK(onerec_params != nullptr);
  const int64_t output_tokens =
      get_onerec_decoder_output_tokens(tokens, params, args_);
  if (output_tokens <= 0) {
    if (onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = false;
    }
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  if (options_.max_tokens_for_graph_mode() > 0 &&
      output_tokens > options_.max_tokens_for_graph_mode()) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "OneRec prefill output token count " << output_tokens
        << " exceeds max_tokens_for_graph_mode ("
        << options_.max_tokens_for_graph_mode()
        << "), falling back to eager mode";
    if (onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = false;
    }
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  torch::Tensor encoder_output;
  if (onerec_params->has_encoder_output) {
    encoder_output = model_->get_onerec_graph_encoder_output();
    if (!encoder_output.defined()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because OneRec encoder output is "
             "not available for decoder prefill ACL graph.";
      if (onerec_params->is_first_prefill) {
        onerec_last_first_prefill_graph_ready_ = false;
      }
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
  }

  const std::string graph_key =
      build_onerec_prefill_graph_key(tokens, positions, params, args_);
  std::lock_guard<std::mutex> graph_lock(onerec_prefill_graph_mutex_);
  if (!onerec_params->is_first_prefill &&
      !onerec_last_first_prefill_graph_ready_) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  auto it = onerec_prefill_graphs_.find(graph_key);
  if (it != onerec_prefill_graphs_.end()) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in OneRec prefill replay mode";
    if (onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = true;
    }
    return it->second->replay(
        model_, tokens, positions, params, encoder_output);
  }

  auto graph =
      std::make_unique<OneRecPrefillAclGraph>(args_, device_, options_);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in OneRec prefill capture mode";
  const bool capture_success = graph->capture(
      model_, tokens, positions, kv_caches, params, encoder_output);
  if (capture_success) {
    LOG(INFO) << "Lazy capturing OneRec prefill ACL graph done, key length: "
              << graph_key.size() << ", output_tokens: " << output_tokens
              << ", is_first_prefill: " << onerec_params->is_first_prefill;
    auto result = graph->output();
    onerec_prefill_graphs_[graph_key] = std::move(graph);
    if (onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = true;
    }
    return result;
  }

  LOG(FATAL) << "Failed to capture OneRec prefill ACL graph, output_tokens: "
             << output_tokens;
  return ModelOutput();
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
ModelOutput AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  const bool in_decoding_phase =
      params_single.meta.batch_forward_type.is_decode();
  const bool in_spec_verify_phase =
      params_single.is_spec_verify &&
      params_single.meta.batch_forward_type.is_chunked_prefill();
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " in_spec_verify_phase: " << in_spec_verify_phase
           << " q_max_seq_len: " << params_single.meta.q_max_seq_len
           << " n_layers: " << args_.n_layers();

  if (is_onerec_decoder_prefill_graph_candidate(
          model_, params_single, options_.enable_onerec_prefill_acl_graph())) {
    return run_onerec_prefill_graph(
        tokens_tensor, positions_tensor, kv_caches, params_single);
  }
  if (const auto* onerec_params = params_single.onerec_params();
      onerec_params != nullptr &&
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL &&
      !onerec_params->is_encoder_forward && onerec_params->is_first_prefill) {
    onerec_last_first_prefill_graph_ready_ = false;
  }

  if ((!in_decoding_phase && !in_spec_verify_phase) || args_.n_layers() == 1) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because the "
           "chunked-prefill validate graph path is currently only adapted for "
           "hybrid linear attention models.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  if (in_decoding_phase &&
      params_single.parallel.dp_global_token_nums.size() > 1) {
    if (params_single.parallel.dp_is_decode.size() !=
        params_single.parallel.dp_global_token_nums.size()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because dp_is_decode size ("
          << params_single.parallel.dp_is_decode.size()
          << ") does not match dp_global_token_nums size ("
          << params_single.parallel.dp_global_token_nums.size()
          << "); ACL graph decode requires valid DP forward metadata. "
          << "dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    if (std::find(params_single.parallel.dp_is_decode.begin(),
                  params_single.parallel.dp_is_decode.end(),
                  0) != params_single.parallel.dp_is_decode.end()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because not all DP ranks are in "
             "decode phase; ACL graph decode requires all DP ranks to be "
             "decode to avoid using prefill or chunked-prefill token counts "
             "as graph bucket size. dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
  }

  // Only use acl graph in decode phase for performance optimization
  // For DP, decode graph bucket should be based on global max tokens across dp
  // groups; local shard can be empty on some ranks.
  uint32_t graph_num_tokens = tokens_tensor.size(/*dim=*/0);
  if (params_single.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params_single.parallel.dp_global_token_nums);
  }
  // Keep actual n_tokens for replay output slicing.
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t local_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t global_batch_size =
      graph_num_tokens / options_.num_decoding_tokens();

  // Large decode batches create too many/too large ACL graphs and may OOM.
  // Fall back to eager mode when batch size exceeds the safety threshold.
  // Use global_batch_size so all DP ranks make the same decision and stay in
  // sync on HCCL collectives.
  const uint32_t decode_batch_size_limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .graph_decode_batch_size_limit()));
  if (global_batch_size > decode_batch_size_limit) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because decode batch_size (global="
        << global_batch_size << ", local=" << local_batch_size << ") > "
        << decode_batch_size_limit
        << "; ACL graph is disabled for this request size to avoid OOM. "
        << "This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = args_.max_position_embeddings();
  const bool seq_len_supported =
      params_single.meta.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because kv_max_seq_len ("
        << params_single.meta.kv_max_seq_len << ") > max_seq_len ("
        << max_seq_len << "). This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint64_t graph_key = get_graph_key(bucket_num_tokens, params_single);

  // Check if captured graph exists for this bucket num_tokens
  auto it = graphs_.find(graph_key);
  if (it != graphs_.end()) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in replay mode";
    auto result = it->second->replay(
        model_, tokens_tensor, positions_tensor, kv_caches, params_single);
    // Handle aux_hidden_states based on options
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(
            result.hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return result;
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph = std::make_unique<AclGraph>(*persistent_param_, device_.index());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in capture mode";
  bool capture_success = false;
  try {
    capture_success = graph->capture(model_,
                                     options_,
                                     tokens_tensor,
                                     positions_tensor,
                                     params_single,
                                     kv_caches,
                                     bucket_num_tokens);
  } catch (const std::exception& e) {
    LOG(ERROR) << "ACL graph capture threw exception for bucket num_tokens="
               << bucket_num_tokens << ": " << e.what()
               << ". Falling back to eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    graphs_[graph_key] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    auto hidden_states = graphs_[graph_key]->get_hidden_states(n_tokens);
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return ModelOutput(hidden_states);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraph::print_graph_tensors() const {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_tokens_: " << persistent_param_.persistent_tokens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_positions_: "
      << persistent_param_.persistent_positions();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_new_cache_slots_: "
      << persistent_param_.persistent_new_cache_slots();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_block_tables_: "
      << persistent_param_.persistent_block_tables();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph hidden_states_: " << persistent_param_.hidden_states();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 16, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

uint64_t AclGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill()) {
    const uint64_t q_max_seq_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

}  // namespace xllm::npu
