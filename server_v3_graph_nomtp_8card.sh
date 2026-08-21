#!/usr/bin/env bash
# NO-MTP baseline (graph on, no speculative decoding).  Identical to
# server_v3_graph_8card.sh except the --draft_model / --num_speculative_tokens /
# --speculative_algorithm=MTP launch params are dropped, so greedy/sampling
# output can be diffed against the MTP-on script to verify spec-verify is
# value-equivalent (no divergence).  _KDA_VERIFY_V3 / GLM5_RMSNORM_ROWWISE match
# the MTP-on script so only "MTP on/off" differs.
set -e
export HCCL_DETERMINISTIC=true
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PROFILING_MODE=dynamic
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export ATB_LOG_LEVEL=ERROR
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.95
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=1
export ATB_COMPARE_TILING_EVERY_KERNEL=0
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
rm -rf /root/atb/log/ /root/ascend/log/ core.*
FLA_NPU_DIR=$(python3 -c "import fla_npu,os;print(os.path.dirname(fla_npu.__file__))" 2>/dev/null)
if [ -n "$FLA_NPU_DIR" ] && [ -f "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash" ]; then
  source "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash"
fi
# V3 fused multi-slot spec-verify is now the code DEFAULT (npu_paged_attention.py
# _KDA_VERIFY_V3 defaults on; GLM5_KDA_VERIFY_V2=1 falls back to legacy V2).
# No env needed to enable V3. Debug/timing switches only:
export GLM5_MTP_TIMING=1
export GLM5_RMSNORM_ROWWISE=1
export XLLM_GRAPH_DEBUG_REJECT=1
export XLLM_GRAPH_REPLAY_TIMING=1

LOCAL_IP=127.0.0.1
PROGRESS_CONN_PORT=9792
MODEL_PATH="/export/home/models/GLM-next-w8a8/"
DRAFT_PATH="/export/home/models/GLM-next-w8a8-mtp"
START_PORT=18994
START_DEVICE=8
CORES_PER_CARD=24
NNODES=8
LOG_DIR=log
COMMUNICATION_BACKEND=hccl
XLLM_PATH="build/xllm/core/server/xllm"
mkdir -p $LOG_DIR
export HCCL_IF_BASE_PORT=43440
MASTER_NODE_ADDR="$LOCAL_IP:$PROGRESS_CONN_PORT"
for (( i=0; i<$NNODES; i++ ))
do
PORT=$((START_PORT + i))
DEVICE=$((START_DEVICE + i))
LOG_FILE="$LOG_DIR/node_$DEVICE.log"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
nohup numactl -C $((DEVICE*CORES_PER_CARD))-$((DEVICE*CORES_PER_CARD+CORES_PER_CARD-1)) $XLLM_PATH \
  --model "$MODEL_PATH" --model_id glm5 --port "$PORT" \
  --master_node_addr="$MASTER_NODE_ADDR" --nnodes="$NNODES" --node_rank="$i" \
  --communication_backend="$COMMUNICATION_BACKEND" --max_memory_utilization=0.80 \
  --enable_chunked_prefill=True --enable_schedule_overlap=True \
  --enable_prefix_cache=False --max_tokens_per_chunk_for_prefill=8192 \
  --enable_mix_batch=False --max_tokens_per_batch=10240 --enable_shm=False \
  \
  --enable_graph=True --model_impl=python \
  --backend=vlm --max_seqs_per_batch=16 --max_body_size=268435456 \
  >> "$LOG_FILE" 2>&1 &
done
echo "V3 graph instance launched (V3=1, V2=0, graph on). master=$MASTER_NODE_ADDR"
