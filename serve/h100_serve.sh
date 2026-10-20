#!/bin/bash
# h100_serve.sh <model_dir> <tag> <workers_per_gpu> <gateway_port> <bridge_port_base> [n_bridges]
# Native (no docker) VLA serving on the H100: workers (sdpa) + gateway + HTTP bridges, logs in /data/jinghui.wang/logs/serve_<tag>/
M=$1; TAG=$2; PER=$3; GW=$4; BP=$5; NB=${6:-8}
P=/data/jinghui.wang/miniconda3/envs/xr1/bin/python; S=/data/jinghui.wang/serve; LD=/data/jinghui.wang/logs/serve_$TAG; mkdir -p $LD
export MIBOT_ATTN=sdpa TOKENIZERS_PARALLELISM=false
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$GW\$"; then echo "gateway port $GW in use"; exit 2; fi
# warm the HF dynamic-module cache (config AND modeling classes) once, before 24+ workers race to (re)write it: a half-written
# transformers_modules/<name>/modeling_mibot.py made workers die with SyntaxError / "source code string cannot contain null bytes"
MIBOT_ATTN=sdpa $P -c "
from transformers import AutoConfig; from transformers.dynamic_module_utils import get_class_from_dynamic_module
c = AutoConfig.from_pretrained('$M', trust_remote_code=True); get_class_from_dynamic_module('modeling_mibot.MiBoTForActionGeneration', '$M'); get_class_from_dynamic_module('processing_mibot.MiBotProcessor', '$M'); print('dynamic modules warm')" 2>/dev/null || echo "warm-up failed (continuing)"
port=$((GW+100)); ports=""
for g in 0 1 2 3 4 5 6 7; do for k in $(seq 1 $PER); do
  CUDA_VISIBLE_DEVICES=$g setsid nohup $P -u $S/server_h100.py --model $M --host 127.0.0.1 --port $port > $LD/worker_$port.log 2>&1 < /dev/null &
  ports="$ports,$port"; port=$((port+1))
done; done
ports=${ports#,}; echo "workers: $ports"
setsid nohup $P -u $S/gateway.py --listen-port $GW --backend-ports $ports --io-timeout 300 --queue-timeout 300 > $LD/gateway.log 2>&1 < /dev/null &
for i in $(seq 0 $((NB-1))); do
  OMP_NUM_THREADS=4 setsid nohup $P -u $S/vla_bridge.py --model $M --port $((BP+i)) --gateway-port $GW --conns 8 > $LD/bridge_$((BP+i)).log 2>&1 < /dev/null &
done
echo "gateway $GW bridges $BP..$((BP+NB-1)) (logs $LD)"
