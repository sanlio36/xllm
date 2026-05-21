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
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>

namespace xllm::debug {

inline bool dsv4_tensor_debug_enabled() {
  const char* value = std::getenv("XLLM_DSV4_DEBUG_TENSOR");
  if (value == nullptr) {
    return false;
  }
  const std::string flag(value);
  return !(flag.empty() || flag == "0" || flag == "false" ||
           flag == "FALSE" || flag == "off" || flag == "OFF");
}

inline bool dsv4_tensor_debug_should_log_layer(int64_t layer_id) {
  const char* value = std::getenv("XLLM_DSV4_DEBUG_LAYER");
  if (value == nullptr || std::string(value).empty()) {
    return layer_id == 0;
  }
  const std::string layer_filter(value);
  if (layer_filter == "all" || layer_filter == "ALL" || layer_filter == "*") {
    return true;
  }
  return layer_id == std::atoll(value);
}

inline void append_tensor_sizes(std::ostringstream& os,
                                const torch::Tensor& tensor) {
  os << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << tensor.size(i);
  }
  os << "]";
}

inline std::string tensor_summary(const std::string& name,
                                  const torch::Tensor& tensor,
                                  int64_t edge_items = 10) {
  std::ostringstream os;
  os << std::setprecision(8);
  os << name << ": ";
  if (!tensor.defined()) {
    os << "undefined";
    return os.str();
  }

  os << "shape=";
  append_tensor_sizes(os, tensor);
  os << ", dtype=" << tensor.scalar_type() << ", device=" << tensor.device()
     << ", numel=" << tensor.numel();
  if (tensor.numel() == 0) {
    os << ", empty";
    return os.str();
  }

  try {
    auto flat = tensor.detach().reshape({-1});
    auto cpu_flat =
        flat.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous();
    const int64_t numel = cpu_flat.numel();
    auto accessor = cpu_flat.accessor<float, 1>();

    os << ", mean=" << cpu_flat.mean().item<float>()
       << ", max=" << cpu_flat.max().item<float>()
       << ", min=" << cpu_flat.min().item<float>();

    const int64_t head_n = std::min<int64_t>(edge_items, numel);
    os << ", head=[";
    for (int64_t i = 0; i < head_n; ++i) {
      if (i > 0) {
        os << ",";
      }
      os << accessor[i];
    }
    os << "]";

    const int64_t tail_n = std::min<int64_t>(edge_items, numel);
    os << ", tail=[";
    for (int64_t i = numel - tail_n; i < numel; ++i) {
      if (i > numel - tail_n) {
        os << ",";
      }
      os << accessor[i];
    }
    os << "]";
  } catch (const c10::Error& e) {
    os << ", summary_error=" << e.what_without_backtrace();
  } catch (const std::exception& e) {
    os << ", summary_error=" << e.what();
  }
  return os.str();
}

inline void log_tensor_summary(const std::string& tag,
                               int64_t layer_id,
                               const std::string& name,
                               const torch::Tensor& tensor) {
  if (!dsv4_tensor_debug_enabled() ||
      !dsv4_tensor_debug_should_log_layer(layer_id)) {
    return;
  }
  std::ostringstream os;
  os << "[DSV4_TENSOR][" << tag << "][layer=" << layer_id << "] "
     << tensor_summary(name, tensor);
  LOG(INFO) << os.str();
}

}  // namespace xllm::debug
