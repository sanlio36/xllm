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

#include "core/framework/config/kernel_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"
#include "core/framework/config/eplb_config.h"

#if defined(USE_NPU)
DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");

DEFINE_int32(enable_fused_mc2,
             -1,
             "Fused MC2 mode for NPU EP MoE. -1 uses auto default, 0 "
             "disables fused MC2, 1 uses DispatchFFNCombine, 2 uses "
             "DispatchGmmCombineDecode.");
#endif

namespace xllm {
namespace {

#if defined(USE_NPU)
int32_t resolve_fused_mc2_mode(int32_t mode) {
  CHECK_GE(mode, -1) << "--enable_fused_mc2 must be -1, 0, 1, or 2.";
  CHECK_LE(mode, 2) << "--enable_fused_mc2 must be -1, 0, 1, or 2.";

  if (mode != -1) {
    return mode;
  }
  if (EPLBConfig::get_instance().expert_parallel_degree() == 2) {
    return 1;
  }
  return 0;
}
#endif

}  // namespace

void KernelConfig::from_flags() {
#if defined(USE_NPU)
  enable_customize_mla_kernel(FLAGS_enable_customize_mla_kernel)
      .npu_kernel_backend(FLAGS_npu_kernel_backend)
      .enable_intralayer_addnorm(FLAGS_enable_intralayer_addnorm)
      .enable_fused_mc2(FLAGS_enable_fused_mc2);
#endif
}

void KernelConfig::from_json(const JsonReader& json) {
#if defined(USE_NPU)
  enable_customize_mla_kernel(
      json.value_or<bool>("enable_customize_mla_kernel",
                          enable_customize_mla_kernel()))
      .npu_kernel_backend(json.value_or<std::string>("npu_kernel_backend",
                                                     npu_kernel_backend()))
      .enable_intralayer_addnorm(json.value_or<bool>(
          "enable_intralayer_addnorm", enable_intralayer_addnorm()))
      .enable_fused_mc2(
          json.value_or<int32_t>("enable_fused_mc2", enable_fused_mc2()));
#endif
}

KernelConfig& KernelConfig::get_instance() {
  static KernelConfig config;
  return config;
}

void KernelConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
#if defined(USE_NPU)
  enable_fused_mc2(resolve_fused_mc2_mode(enable_fused_mc2()));
#endif
}

}  // namespace xllm
