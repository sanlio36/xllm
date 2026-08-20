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

# 确定性计算开关：跨 run 复现 MTP vs 非 MTP 精度对比时必须开启
export HCCL_DETERMINISTIC=true

# MTP 精度修复：NPU reduce tiling 对 S=2 verify 批次与 S=1 decode 走不同路径，
# 在 bf16 舍入边界翻转 1 ULP（首次 L34），经 45 层放大后翻转 argmax（长输出
# 发散/循环根因）。rowwise 使 verify 的 RMSNorm 拆成 S=1 调用，与 decode 锁
# 同一 kernel 路径。验证：1024-token Louis Black greedy 下 MTP == 非 MTP。
export GLM5_RMSNORM_ROWWISE=1

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
PROGRESS_CONN_PORT=9792
MODEL_PATH="/export/home/models/GLM-next-w8a8/"

START_PORT=18994
START_DEVICE=8
CORES_PER_CARD=24
NNODES=8
WORLD_SIZE=8
LOG_DIR=log
COMMUNICATION_BACKEND=hccl
XLLM_PATH="build/xllm/core/server/xllm"
mkdir -p $LOG_DIR
# ========================================

# 旋转量化 checkpoint:ViT 输出进入 LLM 前的隐藏层旋转变换
export HCCL_IF_BASE_PORT=47440

MASTER_NODE_ADDR="$LOCAL_IP:$PROGRESS_CONN_PORT"
LOG_DIR="log"

# 启动服务节点
for (( i=0; i<$NNODES; i++ ))
do
PORT=$((START_PORT + i))
DEVICE=$((START_DEVICE + i))
LOG_FILE="$LOG_DIR/node_$DEVICE.log"
#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
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
--max_tokens_per_chunk_for_prefill=8192 \
--enable_mix_batch=False \
--max_tokens_per_batch=10240 \
--enable_shm=False \
--enable_graph=True \
--model_impl=python \
--backend=vlm \
--draft_model=/export/home/models/GLM-next-w8a8-mtp \
--num_speculative_tokens=1 \
--speculative_algorithm=MTP \
--max_seqs_per_batch=16 \
--max_body_size=268435456 \
)
echo "服务启动命令: ${xllm_command[*]}" >> "$LOG_FILE"
echo "启动节点 $i\(i，设备 npu:\) npu:$DEVICE，端口 $PORT，日志: $LOG_FILE" >> "$LOG_FILE"
"${xllm_command[@]}" >> "$LOG_FILE" 2>&1 &
done

echo "所有节点已启动，主节点地址: $MASTER_NODE_ADDR"
