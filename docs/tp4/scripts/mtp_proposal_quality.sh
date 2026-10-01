#!/usr/bin/env bash
# MTP proposal-quality yardstick (tp2 vs tp4), independent of the verify.
#
# The counting prompt is the point: its continuation is almost perfectly predictable, so a healthy
# MTP head accepts ~98% of its own drafts (measured at tp2). If the tp4 acceptance is far below that
# while the ANSWER is still byte-identical, the proposal head is what is degraded -- the verify and
# the commit path are fine, which is exactly the split this script measures.
#
# Usage: MTP_PROPOSAL_TPS="2 4" bash mtp_proposal_quality.sh
set -uo pipefail

BIN="${BIN:-$HOME/src/ninfer-V100X2-remote/build-v100-tp4/apps/ninfer-serve}"
MODEL="${MODEL:-$HOME/models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer}"
OUT="$HOME/work/ninfer-tp4-m2/proposal-quality-$(date +%m%d-%H%M)"
mkdir -p "$OUT"
PROMPT='继续数列，只输出接下来的 30 个数字，用空格分隔：1 2 3 4 5 6 7 8'

run() { # tp, port
  local tp="$1" port="$2"
  local devs="0,1"; [ "$tp" = 4 ] && devs="0,1,2,3"
  pkill -x ninfer-serve; sleep 6
  nohup env NINFER_FORCE_DIRECT_P2P=1 NINFER_ALLOW_TP4_MTP_UNVERIFIED=1 "$BIN" "$MODEL" \
    --tp "$tp" --devices "$devs" --max-context 8192 --kv-capacity 8192 --host 127.0.0.1 \
    --port "$port" --greedy --no-thinking --spec mtp --draft-tokens 3 --lm-head-draft \
    >"$OUT/tp$tp.log" 2>&1 &
  for _ in $(seq 1 70); do
    sleep 2
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
  done
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"qwen3.8-27b","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":sys.argv[1]}]}))' "$PROMPT")" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]['message']; print('['+'$tp'.strip()+']', ((c.get('reasoning_content') or '')+(c.get('content') or '')).strip()[:160])"
  grep -oE "speculative=mtp [0-9.]+tok/round \([0-9.]+%\)" "$OUT/tp$tp.log" | tail -1
  pkill -x ninfer-serve
}

for tp in ${MTP_PROPOSAL_TPS:-2 4}; do run "$tp" "898$tp"; sleep 2; done

echo "== reference =="
echo "  a healthy MTP head should accept ~98% here; the tp2 figure is the baseline."
echo "  identical answers with a much lower acceptance = proposal-head degradation, not verify."
echo "logdir=$OUT"
