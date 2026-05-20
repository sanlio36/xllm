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

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

#if defined(USE_NPU)
DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");

DEFINE_int32(enable_fused_mc2,
             0,
             "Fused MC2 mode for NPU EP MoE. 0 disables fused MC2, 1 uses "
             "DispatchFFNCombine, 2 uses DispatchGmmCombineDecode.");

DEFINE_bool(
    enable_fused_moe_gmm_swiglu,
    false,
    "Whether to fuse W8A8 MoE grouped matmul1 and dequant+swiglu+quant.");
#endif

namespace xllm {

void KernelConfig::from_flags() {
#if defined(USE_NPU)
  enable_customize_mla_kernel(FLAGS_enable_customize_mla_kernel)
      .npu_kernel_backend(FLAGS_npu_kernel_backend)
      .enable_intralayer_addnorm(FLAGS_enable_intralayer_addnorm)
      .enable_fused_mc2(FLAGS_enable_fused_mc2)
      .enable_fused_moe_gmm_swiglu(FLAGS_enable_fused_moe_gmm_swiglu);
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
          json.value_or<int32_t>("enable_fused_mc2", enable_fused_mc2()))
      .enable_fused_moe_gmm_swiglu(json.value_or<bool>(
          "enable_fused_moe_gmm_swiglu", enable_fused_moe_gmm_swiglu()));
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
}

}  // namespace xllm
