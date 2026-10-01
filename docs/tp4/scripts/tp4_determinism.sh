#!/usr/bin/env bash
# Is the tp4 decode path deterministic, and is the captured graph numerically equal to the eager
# path? Greedy sampling makes both questions decidable by byte comparison.
#
#   4 passes: [graph, graph, eager, eager] x [prompt set], all no-spec.
#   - pass0 vs pass1 equal  -> the graph path is deterministic
#   - pass2 vs pass3 equal  -> the eager path is deterministic
#   - pass0 vs pass2 equal  -> capture does not change the arithmetic
set -uo pipefail

BIN="$HOME/src/ninfer-V100X2-remote/build-v100-tp4/apps/ninfer-serve"
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer}"
OUT="$HOME/work/ninfer-tp4-m2/determinism-$(date +%m%d-%H%M)"
mkdir -p "$OUT"

run_pass() { # port, tag, extra...
  local port="$1" tag="$2"; shift 2
  NINFER_FORCE_DIRECT_P2P=1 NINFER_ALLOW_TP4_MTP_UNVERIFIED=1 nohup "$BIN" "$MODEL" \
    --tp "${TP:-4}" --devices "${GPUS:-0,1,2,3}" --max-context 8192 --kv-capacity 8192 \
    --host 127.0.0.1 --port "$port" --greedy --no-thinking "$@" >"$OUT/serve-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 150); do
    sleep 2
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "DIED $tag"; return 1; }
  done
  curl -sf "http://127.0.0.1:$port/health" >/dev/null || { echo "TIMEOUT $tag"; return 1; }
  python3 - "$port" "$OUT" "$tag" <<'PY'
import json, sys, urllib.request
port, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "我养了两只猫，老大叫橘子，是狸花，怕水；老二叫煤球，是黑猫，胆子大。请问胆子大的那只是什么花色？",
    "一个房间里没椅子，先搬进来两把，又搬走一把，现在有几把？只回答数字。",
    "把下面这句话翻译成英文：山上有座庙，庙里有个老和尚。",
    "用一句话说明为什么 0.1 + 0.2 != 0.3。",
]
for i, p in enumerate(prompts):
    body = json.dumps({"model": "qwen3.8-27b", "max_tokens": 96, "temperature": 0,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    c = d["choices"][0]["message"]
    open(f"{out}/{tag}-{i}.txt", "w").write((c.get("reasoning_content") or "") + (c.get("content") or ""))
    print(f"  {tag} prompt {i}: {d['usage']['completion_tokens']} tok")
PY
  kill $pid 2>/dev/null
  for _ in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  for _ in $(seq 1 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    (( free > 15000 )) && break
    sleep 1
  done
}

echo "== free the cards =="
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8901 >/dev/null
bash "$HOME/scripts/ninfer-v3/serve_v100x2_v2.sh" stop 8902 >/dev/null
sleep 3
nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' '; echo

run_pass 8920 g0
run_pass 8920 g1
run_pass 8920 e0 --no-cuda-graph
run_pass 8920 e1 --no-cuda-graph

echo "== verdict =="
for i in 0 1 2 3; do
  g=OK; e=OK; ge=OK
  cmp -s "$OUT/g0-$i.txt" "$OUT/g1-$i.txt" || g=NONDET
  cmp -s "$OUT/e0-$i.txt" "$OUT/e1-$i.txt" || e=NONDET
  cmp -s "$OUT/g0-$i.txt" "$OUT/e0-$i.txt" || ge=DIFFERS
  echo "  prompt $i: graph=$g eager=$e graph-vs-eager=$ge"
  if [[ "$ge" == DIFFERS ]]; then
    echo "    graph: $(head -c 90 "$OUT/g0-$i.txt" | tr '\n' ' ')"
    echo "    eager: $(head -c 90 "$OUT/e0-$i.txt" | tr '\n' ' ')"
  fi
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
