#!/bin/bash
# TP2 regression gate: start the given serve binary on a TP2 config, send one greedy request, stop it.
# Usage: tp2_gate_serve_ab.sh <binary> <port> <tag>   (compare the JSON/KV/graph-node lines between tags)
# 用法: serve_ab.sh <binary> <port> <tag>
BIN=$1; PORT=$2; TAG=$3
ART=$HOME/models/ninfer-v2-official/qwen3_8_27b_nvfp4.ninfer
LOG=/tmp/ab_${TAG}.log
export NINFER_FORCE_DIRECT_P2P=1
cd ~/src/ninfer-V100X2-remote
nohup env NINFER_FORCE_DIRECT_P2P=1 $BIN $ART --tp 2 --devices 0,1 \
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 --prefill-chunk 4096 \
  --pending-timeout-ms 600000 --spec mtp --draft-tokens 3 --lm-head-draft \
  --host 127.0.0.1 --port $PORT > $LOG 2>&1 &
PID=$!
for i in $(seq 1 90); do
  ss -tln 2>/dev/null | grep -q ":$PORT " && break
  sleep 5
  kill -0 $PID 2>/dev/null || { echo "$TAG: 进程死了"; tail -5 $LOG; exit 1; }
done
ss -tln 2>/dev/null | grep -q ":$PORT " || { echo "$TAG: 起不来"; tail -5 $LOG; exit 1; }
echo "=== $TAG ready (pid $PID) ==="
grep -E "direct P2P|probe" $LOG | head -3
T0=$(date +%s.%N)
RESP=$(curl -s -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with exactly: alpha beta gamma"}],"temperature":0,"max_tokens":64}')
T1=$(date +%s.%N)
echo "$RESP" > /tmp/ab_${TAG}.json
python3 - "$TAG" <<'PY'
import json,sys
d=json.load(open(f"/tmp/ab_{sys.argv[1]}.json"))
c=d["choices"][0]
print("content:", repr(c["message"]["content"]))
print("finish:", c.get("finish_reason"), "usage:", d.get("usage"))
PY
echo "wall: $(echo "$T1 - $T0" | bc) s"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
kill $PID; for i in $(seq 1 30); do sleep 2; kill -0 $PID 2>/dev/null || break; done
echo "=== $TAG stopped ==="
