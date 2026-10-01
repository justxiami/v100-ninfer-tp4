#!/usr/bin/env bash
# tp4 MTP differential probe.
#
# Greedy speculative decoding is EXACT: with `--greedy` the tp4 no-spec answer and the tp4 MTP
# answer must be byte-identical. Any difference is a real defect in the round (not a sampling
# artifact), and its size tells us how early the divergence starts.
#
# Also sweeps --draft-tokens 1 vs 3: if the accept rate is broken at k=1 too, the fault is in the
# single-token proposal/verify path rather than in the multi-step AR chain.
set -uo pipefail

BIN="$HOME/src/ninfer-V100X2-remote/build-v100-tp4/apps/ninfer-serve"
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer}"
TAG="${TAG:-$(basename "$MODEL" .ninfer)}-tp${TP:-4}"
LOGDIR="$HOME/work/ninfer-tp4-m2/diff-$(date +%m%d-%H%M)-$TAG"
mkdir -p "$LOGDIR"
MAXTOK="${MAXTOK:-256}"

start() {
  local port="$1"; shift
  local extra=()
  [[ "${NOGRAPH:-0}" == 1 ]] && extra+=(--no-cuda-graph)
  NINFER_FORCE_DIRECT_P2P=1 NINFER_ALLOW_TP4_MTP_UNVERIFIED=1 nohup "$BIN" "$MODEL" \
    --tp "${TP:-4}" --devices "${GPUS:-0,1,2,3}" --max-context 8192 --kv-capacity 8192 --host 127.0.0.1 \
    --port "$port" --greedy --no-thinking "${extra[@]}" "$@" >"$LOGDIR/serve-$port.log" 2>&1 &
  echo $! > "$LOGDIR/pid-$port"
  for _ in $(seq 1 150); do
    sleep 2
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { echo "UP $port"; return 0; }
    kill -0 "$(cat "$LOGDIR/pid-$port")" 2>/dev/null || { echo "DIED $port"; return 1; }
  done
  echo "TIMEOUT $port"; return 1
}

stop() {
  local port="$1"
  [[ -f "$LOGDIR/pid-$port" ]] || return 0
  kill "$(cat "$LOGDIR/pid-$port")" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$(cat "$LOGDIR/pid-$port")" 2>/dev/null || break; sleep 1; done
  for _ in $(seq 1 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    (( free > 15000 )) && break
    sleep 1
  done
}

ask() { # port, outfile, prompt-index
  local port="$1" out="$2" idx="$3"
  python3 - "$port" "$out" "$idx" "$MAXTOK" <<'PY'
import json, sys, urllib.request
port, out, idx, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompts = [
    "我养了两只猫，老大叫橘子，是狸花，怕水；老二叫煤球，是黑猫，胆子大。请问胆子大的那只是什么花色？",
    "一个房间里没椅子，先搬进来两把，又搬走一把，现在有几把？只回答数字。",
    "把下面这句话翻译成英文：山上有座庙，庙里有个老和尚。",
    "用一句话说明为什么 0.1 + 0.2 != 0.3。",
]
body = json.dumps({"model": "qwen3.8-27b", "max_tokens": maxtok, "temperature": 0,
                   "messages": [{"role": "user", "content": prompts[idx]}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=600))
c = d["choices"][0]["message"]
text = (c.get("reasoning_content") or "") + (c.get("content") or "")
usage = d.get("usage", {})
open(out, "w").write(text)
print(f"[{idx}] {len(text)} chars, completion_tokens={usage.get('completion_tokens')}")
PY
}

scan() { # port, tag
  local port="$1" tag="$2"
  for i in 0 1 2 3; do
    ask "$port" "$LOGDIR/$tag-$i.txt" "$i"
  done
  grep -oE "speculative=[a-z]+ [0-9.]+tok/round \([0-9.]+%\)" "$LOGDIR/serve-$port.log" | sort | uniq -c
}

echo "== free the cards =="
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8901 >/dev/null
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8902 >/dev/null
sleep 3
nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' '; echo

echo "== tp4 greedy, no spec =="
if start 8915; then scan 8915 nospec; else tail -3 "$LOGDIR/serve-8915.log"; fi
stop 8915

for K in 1 3; do
  echo "== tp4 greedy, mtp k=$K =="
  if start 8916 --spec mtp --draft-tokens "$K" --lm-head-draft; then
    scan 8916 "mtp$K"
  else
    tail -3 "$LOGDIR/serve-8916.log"
  fi
  stop 8916
done

if [[ "${NOGRAPH:-0}" == 1 ]]; then
  echo "== tp4 greedy, mtp k=1, NO CUDA GRAPH =="
  if start 8917 --spec mtp --draft-tokens 1 --lm-head-draft; then scan 8917 mtp1ng; else tail -3 "$LOGDIR/serve-8917.log"; fi
  stop 8917
fi

echo "== diff (no-spec is the reference; any difference is a defect) =="
for i in 0 1 2 3; do
  for K in 1 3 ${NOGRAPH:+1ng}; do
    if [[ -f "$LOGDIR/mtp$K-$i.txt" ]]; then
      if cmp -s "$LOGDIR/nospec-$i.txt" "$LOGDIR/mtp$K-$i.txt"; then
        echo "  prompt $i k=$K: IDENTICAL"
      else
        first=$(cmp "$LOGDIR/nospec-$i.txt" "$LOGDIR/mtp$K-$i.txt" 2>/dev/null | sed 's/.*byte \([0-9]*\).*/\1/')
        echo "  prompt $i k=$K: DIFFERS at byte ${first:-?} (nospec=$(wc -c <"$LOGDIR/nospec-$i.txt")B mtp=$(wc -c <"$LOGDIR/mtp$K-$i.txt")B)"
        python3 - "$LOGDIR/nospec-$i.txt" "$LOGDIR/mtp$K-$i.txt" <<'PY'
import sys, difflib
a, b = (open(p).read() for p in sys.argv[1:3])
for line in list(difflib.unified_diff([a], [b], "nospec", "mtp", n=0))[2:4]:
    print("    " + line[:300])
PY
      fi
    fi
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
echo "restored: 8901=$a 8902=$b"
echo "logdir=$LOGDIR"
