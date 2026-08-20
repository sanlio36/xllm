/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/py_hyper_connection.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace py = pybind11;

namespace xllm {

void register_hyper_connection_kernels(py::module_& m) {
  // hc_pre: fused rsqrt + linear + sinkhorn + weighted-sum-reduce.
  // x: [B, S, hc_mult, D] — the full 4-D residual stream cube.
  // Returns (output [B,S,D], post [B,S,hc_mult], comb [B,S,hc_mult,hc_mult]).
  m.def("hc_pre",
        &kernel::npu::hc_pre,
        py::arg("x"),
        py::arg("hc_fn"),
        py::arg("hc_scale"),
        py::arg("hc_base"),
        py::arg("hc_mult"),
        py::arg("hc_sinkhorn_iters"),
        py::arg("norm_eps"),
        py::arg("hc_eps"));

  // hc_post: post * x + comb^T @ residual (fused recombination).
  // x: [B, S, D], residual: [B, S, hc_mult, D],
  // post: [B, S, hc_mult], comb: [B, S, hc_mult, hc_mult].
  // Returns output [B, S, hc_mult, D].
  m.def("hc_post",
        &kernel::npu::hc_post,
        py::arg("x"),
        py::arg("residual"),
        py::arg("post"),
        py::arg("comb"));
}

}  // namespace xllm