#!/usr/bin/env bash
# Minimal tp4 MTP crash repro at load-time warmup.
#
# The load-time warmup runs one representative MTP round per batch size (prepare_graphs), so a
# crash there needs no request at all: start the server, wait, and read the log. Runs N attempts
# because the fault looked intermittent (an out-of-bounds write that sometimes lands in mapped
# memory).
set -uo pipefail

BIN="$HOME/src/ninfer-V100X2-remote/build-v100-tp4/apps/ninfer-serve"
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer}"
OUT="$HOME/work/ninfer-tp4-m2/repro-$(date +%m%d-%H%M)"
mkdir -p "$OUT"
ATTEMPTS="${ATTEMPTS:-3}"
PORT="${PORT:-8930}"

bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8901 >/dev/null
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8902 >/dev/null
sleep 3

for n in $(seq 1 "$ATTEMPTS"); do
  for K in ${KS:-3 1}; do
    log="$OUT/attempt-$n-k$K.log"
    NINFER_FORCE_DIRECT_P2P=1 NINFER_ALLOW_TP4_MTP_UNVERIFIED=1 \
      timeout 300 "$BIN" "$MODEL" --tp 4 --devices 0,1,2,3 --max-context 8192 \
      --kv-capacity 8192 --host 127.0.0.1 --port "$PORT" --greedy \
      --spec mtp --draft-tokens "$K" --lm-head-draft ${EXTRA:-} >"$log" 2>&1 &
    pid=$!
    # Warmup either finishes (listening) or faults; 150s covers a full 4-GPU load at ~28s/load.
    for _ in $(seq 1 75); do
      sleep 2
      grep -q "listening on" "$log" && break
      kill -0 $pid 2>/dev/null || break
    done
    if grep -q "listening on" "$log"; then verdict="OK"; else verdict="FAIL: $(grep -oE 'failed: [A-Za-z]+' "$log" | tail -1)"; fi
    echo "attempt $n k=$K -> $verdict"
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    for _ in $(seq 1 60); do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
      (( free > 15000 )) && break
      sleep 1
    done
  done
done

echo "== restore served instances =="
MODEL="$HOME/models/ninfer-V100X2/qwen3_8_27b_w4a4w8a8.ninfer" GPUS=2,3 KV_CAPACITY=131072 \
  bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" start 8901 >/dev/null
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" start 8902 >/dev/null
for _ in $(seq 1 60); do
  sleep 3
  a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8901/health)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8902/health)
  [[ "$a" == 200 && "$b" == 200 ]] && break
done
echo "restored: 8901=$a 8902=$b  out=$OUT"
