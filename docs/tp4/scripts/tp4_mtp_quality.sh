#!/usr/bin/env bash
# tp4 quality smoke, no-spec vs MTP, same weights, same prompts.
#
# Why this script exists: the tp4 MTP round used to crash, then ran but produced garbage. It stops
# the two SERVED instances (8901/8902) to free all four cards, runs both tp4 configurations on a
# scratch port, prints each answer and the round statistics, then restores the served instances.
# Restoring is not optional: 8901/8902 are the two live panels in llm-hud.
set -uo pipefail

BIN="$HOME/src/ninfer-V100X2-remote/build-v100-tp4/apps/ninfer-serve"
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer}"
TAG="${TAG:-$(basename "$MODEL" .ninfer)}"
LOGDIR="$HOME/work/ninfer-tp4-m2/quality-$(date +%m%d-%H%M)-$TAG"
mkdir -p "$LOGDIR"

start() { # port, extra args...
  local port="$1"; shift
  NINFER_FORCE_DIRECT_P2P=1 NINFER_ALLOW_TP4_MTP_UNVERIFIED=1 nohup "$BIN" "$MODEL" --tp 4 --devices 0,1,2,3 \
    --max-context 8192 --kv-capacity 8192 --host 127.0.0.1 --port "$port" \
    "$@" >"$LOGDIR/serve-$port.log" 2>&1 &
  echo $! > "$LOGDIR/pid-$port"
  for _ in $(seq 1 120); do
    sleep 2
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then echo "UP $port"; return 0; fi
    if ! kill -0 "$(cat "$LOGDIR/pid-$port")" 2>/dev/null; then echo "DIED $port"; return 1; fi
  done
  echo "TIMEOUT $port"; return 1
}

stop() {
  local port="$1"
  [[ -f "$LOGDIR/pid-$port" ]] || return 0
  kill "$(cat "$LOGDIR/pid-$port")" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$(cat "$LOGDIR/pid-$port")" 2>/dev/null || break
    sleep 1
  done
  # Memory is not returned the instant the process exits; the next start needs it back.
  for _ in $(seq 1 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    (( free > 15000 )) && break
    sleep 1
  done
  echo "free-after-stop=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"
}

ask() { # port, tag, prompt
  local port="$1" tag="$2" prompt="$3"
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"qwen3.8-27b","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":400,"temperature":0}))' "$prompt")" \
    > "$LOGDIR/answer-$tag.json"
  python3 - "$LOGDIR/answer-$tag.json" "$tag" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    c = d["choices"][0]["message"]
    text = (c.get("reasoning_content") or "")
    text += c.get("content") or ""
    print(f"--- {sys.argv[2]} ---")
    print(text.strip()[:1200])
except Exception as e:
    print(f"--- {sys.argv[2]} --- PARSE FAIL {e}: {open(sys.argv[1]).read()[:400]}")
PY
}

P1='我养了两只猫，老大叫橘子，是狸花，怕水；老二叫煤球，是黑猫，胆子大。请问胆子大的那只是什么花色？'
P2='一个房间里没椅子，先搬进来两把，又搬走一把，现在有几把？只回答数字。'

echo "== stop served instances =="
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8901
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8902
sleep 3
echo "free-after-stop=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"

echo "== tp4 no-spec =="
if start 8915; then ask 8915 nospec-p1 "$P1"; ask 8915 nospec-p2 "$P2"; fi
grep -E "tok/round|graph-nodes" "$LOGDIR/serve-8915.log" | tail -5
stop 8915

echo "== tp4 mtp =="
if start 8916 --spec mtp --draft-tokens 3 --lm-head-draft; then
  ask 8916 mtp-p1 "$P1"; ask 8916 mtp-p2 "$P2"
fi
grep -E "tok/round|graph-nodes" "$LOGDIR/serve-8916.log" | tail -8
stop 8916

echo "== restore served instances =="
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer}"
TAG="${TAG:-$(basename "$MODEL" .ninfer)}" GPUS=2,3 KV_CAPACITY=131072 \
  bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" start 8901
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" start 8902
for _ in $(seq 1 60); do
  sleep 3
  a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8901/health)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8902/health)
  [[ "$a" == 200 && "$b" == 200 ]] && break
done
echo "restored: 8901=$a 8902=$b"
echo "logdir=$LOGDIR"
