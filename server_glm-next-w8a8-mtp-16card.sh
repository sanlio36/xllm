#!/bin/bash
set -e

# 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PROFILING_MODE=dynamic

# Ascend 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 日志和性能配置
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.95
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

# 并行配置
export OMP_NUM_THREADS=12


# HCCL 配置
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

# 清理日志和核心文件
rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*

FLA_NPU_DIR=$(python3 -c "import fla_npu,os;print(os.path.dirname(fla_npu.__file__))" 2>/dev/null)
if [ -n "$FLA_NPU_DIR" ] && [ -f "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash" ]; then
    source "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash"
    echo "[launch] sourced fla_npu OPP env from $FLA_NPU_DIR"
else
    echo "[launch] WARNING: fla_npu OPP set_env.bash not found; KDA 561103 likely"
fi


# ============ 参数 ============
LOCAL_IP=127.0.0.1
PROGRESS_CONN_PORT=9592
MODEL_PATH="/export/home/models/GLM-next-w8a8/"

START_PORT=18994
START_DEVICE=0
CORES_PER_CARD=24
NNODES=16
WORLD_SIZE=16
LOG_DIR=log
COMMUNICATION_BACKEND=hccl
XLLM_PATH="build/xllm/core/server/xllm"
mkdir -p $LOG_DIR
# ========================================

# 旋转量化 checkpoint:ViT 输出进入 LLM 前的隐藏层旋转变换
export HCCL_IF_BASE_PORT=43440

MASTER_NODE_ADDR="$LOCAL_IP:$PROGRESS_CONN_PORT"
LOG_DIR="log"

# 启动服务节点
for (( i=0; i<$NNODES; i++ ))
do
PORT=$((START_PORT + i))
DEVICE=$((START_DEVICE + i))
LOG_FILE="$LOG_DIR/node_$DEVICE.log"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
xllm_command=(
nohup numactl -C $((DEVICE*CORES_PER_CARD))-$((DEVICE*CORES_PER_CARD+CORES_PER_CARD-1)) $XLLM_PATH \
--model "$MODEL_PATH" \
--model_id glm5 \
--port "$PORT" \
--master_node_addr="$MASTER_NODE_ADDR" \
--nnodes="$NNODES" \
--node_rank="$i" \
--communication_backend="$COMMUNICATION_BACKEND" \
--max_memory_utilization=0.85 \
--enable_chunked_prefill=False \
--enable_schedule_overlap=False \
--enable_prefix_cache=False \
--enable_shm=False \
--model_impl=python \
--backend=llm \
--draft_model=/export/home/models/GLM-next-w8a8-mtp \
--num_speculative_tokens=1 \
--speculative_algorithm=MTP \
--max_seqs_per_batch=16 \
)
echo "服务启动命令: ${xllm_command[*]}" >> "$LOG_FILE"
echo "启动节点 $i\(i，设备 npu:\) npu:$DEVICE，端口 $PORT，日志: $LOG_FILE" >> "$LOG_FILE"
"${xllm_command[@]}" >> "$LOG_FILE" 2>&1 &
done

echo "所有节点已启动，主节点地址: $MASTER_NODE_ADDR"
