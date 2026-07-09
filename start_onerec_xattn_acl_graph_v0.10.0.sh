#!/usr/bin/env bash
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

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the OneRec model directory.}"
: "${XLLM_BIN:?Set XLLM_BIN to the xllm server executable.}"

ASCEND_TOOLKIT_ENV=${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
DEVICE=${DEVICE:-0}
PORT=${PORT:-9438}
XLLM_BEAM_WIDTH=${XLLM_BEAM_WIDTH:-256}
MODEL_ID=${MODEL_ID:-t5_01B}
MAX_TOKENS_FOR_GRAPH_MODE=${MAX_TOKENS_FOR_GRAPH_MODE:-20000}
ENABLE_CONSTRAINED_DECODING=${ENABLE_CONSTRAINED_DECODING:-true}
LOG_DIR=${LOG_DIR:-logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/onerec_xattn_acl_graph_${PORT}.log}

for env_file in "$ASCEND_TOOLKIT_ENV" "$ATB_ENV"; do
  if [[ ! -f "$env_file" ]]; then
    echo "Environment script not found: $env_file" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$env_file"
done

if [[ ! -x "$XLLM_BIN" ]]; then
  echo "XLLM_BIN is not executable: $XLLM_BIN" >&2
  exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if [[ -n "${TCMALLOC_PATH:-}" ]]; then
  if [[ ! -f "$TCMALLOC_PATH" ]]; then
    echo "TCMALLOC_PATH does not exist: $TCMALLOC_PATH" >&2
    exit 1
  fi
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${TCMALLOC_PATH}"
fi

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export NPU_MEMORY_FRACTION=${NPU_MEMORY_FRACTION:-0.98}
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=${ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE:-3}
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=${ATB_WORKSPACE_MEM_ALLOC_GLOBAL:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export INF_NAN_MODE_FORCE_DISABLE=${INF_NAN_MODE_FORCE_DISABLE:-1}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-11532}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export FOLLY_DEBUG_MEMORYIDLER_DISABLE_UNMAP=${FOLLY_DEBUG_MEMORYIDLER_DISABLE_UNMAP:-1}
export GLOG_minloglevel=${GLOG_minloglevel:-0}

ASCEND_RT_VISIBLE_DEVICES="$DEVICE" EXTRA_THREAD_NUM=${EXTRA_THREAD_NUM:-16} \
  "$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --model_id "$MODEL_ID" \
  --backend=rec \
  --port "$PORT" \
  --max_memory_utilization=0.5 \
  --prefill_scheduling_memory_usage_threshold=0.5 \
  --max_tokens_per_batch=20000 \
  --max_seqs_per_batch=20 \
  --block_size=128 \
  --ep_size=1 \
  --dp_size=1 \
  --enable_rec_prefill_only=false \
  --max_decode_rounds=3 \
  --beam_width="$XLLM_BEAM_WIDTH" \
  --enable_prefix_cache=false \
  --enable_schedule_overlap=false \
  --enable_chunked_prefill=false \
  --enable_constrained_decoding="$ENABLE_CONSTRAINED_DECODING" \
  --enable_output_sku_logprobs=false \
  --enable_convert_tokens_to_item=false \
  --enable_graph=true \
  --enable_onerec_prefill_acl_graph=true \
  --max_tokens_for_graph_mode="$MAX_TOKENS_FOR_GRAPH_MODE" \
  --rec_worker_max_concurrency=2 \
  "$@" \
  >"$LOG_FILE" 2>&1 &

echo "xLLM started with PID $!. Log: $LOG_FILE"
