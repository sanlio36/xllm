#!/bin/bash
# 验证 MTP 并发输出 == 单并发出（修复 RMSNorm rowwise cap 4->64 +
# _UnweightedRMSNorm rowwise 后）。无 tensor dump，纯 char 级对比。
#
# 流程：起 MTP (graph=eager可选) -> 单发1条(max_tokens=1024,T=0)->base ->
#   干净重启 -> 3并发 -> conc{0,1,2} -> 逐字符对比 base -> result 文件。
#
# 用法: bash tools/mtp_conc_validate.sh [GRAPH] [START_DEVICE]
#   GRAPH       true|false  (默认 true=图模式，用户最终目标)
#   START_DEVICE 8|0        (默认 8)
set -uo pipefail
cd /export/home/nielinfeng/xllm

GRAPH="${1:-true}"
START_DEVICE="${2:-8}"
NNODES=8
LOG=/tmp/dump/conc_validate.log
RESULT=/tmp/dump/conc_validate_result.txt
REQ=/tmp/dump/long_request.json          # max_tokens=1024, T=0
PORT=18994
BASE_PORT=47440
PROGRESS_CONN_PORT=9792
MODEL_PATH="/export/home/models/GLM-next-w8a8/"
DRAFT="/export/home/models/GLM-next-w8a8-mtp"
CORES_PER_CARD=24
DEVS=$(seq -s, $START_DEVICE $((START_DEVICE+NNODES-1)))
mkdir -p /tmp/dump
: > "$LOG"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "GRAPH=$GRAPH START_DEVICE=$START_DEVICE VISIBLE=$DEVS"

launch_server(){
  pkill -9 -f 'build/xllm/core/server/xllm' 2>/dev/null; sleep 8
  pkill -9 -f 'build/xllm/core/server/xllm' 2>/dev/null; sleep 5
  rm -rf log/node_*.log
  export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
  export PYTHON_LIB_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
  export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch,os;print(os.path.dirname(os.path.abspath(torch.__file__)))')"
  export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
  export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export PROFILING_MODE=dynamic
  # Ascend/atb set_env.sh 引用未定义变量 (ZSH_VERSION 等)，set -u 下会退出
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export HCCL_DETERMINISTIC=true
  export GLM5_RMSNORM_ROWWISE=1
  export ASDOPS_LOG_TO_STDOUT=0 ASDOPS_LOG_LEVEL=ERROR ATB_LOG_TO_STDOUT=1
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NPU_MEMORY_FRACTION=0.95
  export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
  export OMP_NUM_THREADS=12
  export HCCL_CONNECT_TIMEOUT=7200 HCCL_OP_EXPANSION_MODE="AIV"
  rm -rf /root/atb/log/ /root/ascend/log/ core.*
  FLA_NPU_DIR=$(python3 -c "import fla_npu,os;print(os.path.dirname(fla_npu.__file__))" 2>/dev/null)
  [ -n "$FLA_NPU_DIR" ] && [ -f "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash" ] \
    && source "$FLA_NPU_DIR/opp/vendors/fla_npu_transformer/bin/set_env.bash"
  set -u
  export HCCL_IF_BASE_PORT=$BASE_PORT
  MASTER_NODE_ADDR="127.0.0.1:$PROGRESS_CONN_PORT"
  for (( i=0; i<NNODES; i++ )); do
    PORT_i=$((PORT+i)); DEVICE=$((START_DEVICE+i))
    ASCEND_RT_VISIBLE_DEVICES=$DEVS numactl -C $((DEVICE*CORES_PER_CARD))-$((DEVICE*CORES_PER_CARD+CORES_PER_CARD-1)) \
      build/xllm/core/server/xllm \
      --model "$MODEL_PATH" --model_id glm5 --port "$PORT_i" \
      --master_node_addr="$MASTER_NODE_ADDR" --nnodes=$NNODES --node_rank=$i \
      --communication_backend=hccl --max_memory_utilization=0.85 \
      --enable_chunked_prefill=False --enable_schedule_overlap=False \
      --enable_prefix_cache=False --max_tokens_per_chunk_for_prefill=8192 \
      --enable_mix_batch=False --max_tokens_per_batch=10240 --enable_shm=False \
      --enable_graph="$GRAPH" --model_impl=python --backend=vlm \
      --draft_model="$DRAFT" --num_speculative_tokens=1 --speculative_algorithm=MTP \
      --max_seqs_per_batch=16 --max_body_size=268435456 \
      >> log/node_$DEVICE.log 2>&1 &
  done
}

wait_ready(){
  local n=0
  while ! curl -sS -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
    n=$((n+1)); [ $n -gt 240 ] && { log "启动超时(480s)"; tail -20 log/node_$START_DEVICE.log; return 1; }; sleep 2
  done
  sleep 8
  local ok=0 t r
  for t in 1 2 3; do
    r=$(curl -sS -m 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | head -c 50)
    [ -n "$r" ] && ok=$((ok+1)) || ok=0; [ $ok -ge 2 ] && break; sleep 3
  done
  [ $ok -ge 2 ] || { log "服务假死"; return 1; }
  log "服务就绪 (~$((n*2))s)"; return 0
}

stop_server(){ pkill -9 -f 'build/xllm/core/server/xllm' 2>/dev/null; sleep 10; }

# ---------- 单并发 base ----------
log "=== Run 1: 单并发 (base) ==="
launch_server || { log "启动失败"; exit 1; }
wait_ready || exit 1
python3 - "$REQ" <<'PY'
import json,sys,urllib.request
req=json.load(open(sys.argv[1])); d=json.dumps(req).encode()
r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18994/v1/chat/completions',d,{'Content-Type':'application/json'}),timeout=900)
c=json.load(r)['choices'][0]['message']['content']
json.dump(c,open('/tmp/dump/val_base.txt','w'))
print("base chars=",len(c))
PY
log "单并发完成"; stop_server

# ---------- 3 并发 ----------
log "=== Run 2: 3 并发 ==="
launch_server || { log "启动失败"; exit 1; }
wait_ready || exit 1
python3 - "$REQ" <<'PY'
import json,sys,urllib.request,concurrent.futures as cf
req=json.load(open(sys.argv[1])); d=json.dumps(req).encode()
def call(i):
    try:
        r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18994/v1/chat/completions',d,{'Content-Type':'application/json'}),timeout=900)
        return json.load(r)['choices'][0]['message']['content']
    except Exception as e:
        return f"__ERR__:{e}"
with cf.ThreadPoolExecutor(3) as ex: outs=list(ex.map(call,range(3)))
for i,c in enumerate(outs): json.dump(c,open(f'/tmp/dump/val_conc_{i}.txt','w'))
print("conc done")
PY
log "并发完成"; stop_server

# ---------- 对比 ----------
python3 - <<'PY'
import json
base=json.load(open('/tmp/dump/val_base.txt'))
lines=[f"base={len(base)}"]
all_same=True
for i in range(3):
    try: c=json.load(open(f'/tmp/dump/val_conc_{i}.txt'))
    except Exception as e: c=""
    if c.startswith("__ERR__:"):
        lines.append(f"conc{i}=ERR {c[:60]}"); all_same=False; continue
    same = (base==c)
    n=min(len(base),len(c)); j=0
    while j<n and base[j]==c[j]: j+=1
    if not same: all_same=False
    lines.append(f"conc{i}={len(c)} {'SAME' if same else f'DIVERGE@{j}'}")
res="\n".join(lines)
open('/tmp/dump/conc_validate_result.txt','w').write(res+"\n")
print(res)
print("\n>>> ALIGN (并发==单并发)" if all_same else "\n>>> STILL DIVERGES")
PY
log "验证结束，结果见 $RESULT"
