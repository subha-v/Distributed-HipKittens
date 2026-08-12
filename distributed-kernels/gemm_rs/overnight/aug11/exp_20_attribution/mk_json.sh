#!/usr/bin/env bash
# aug11/exp_20: build ablation.json from the reattribute log, and print it back.
# CPU-only; safe to run while a GPU job is live.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
LOG=$(ls -t $ON/experiments/logs/reattribute_*.log 2>/dev/null | head -1)
echo "source log: $LOG"

# Compact clock summary: the distinct sclk levels reported and the perf level.
summarize() {  # summarize <clocks file>
  local f=$1
  [ -f "$f" ] || { echo "not captured"; return; }
  local sclk lvl stamp
  stamp=$(head -1 "$f")
  # The level id is a digit under load but the literal 'S' when the GPUs have
  # dropped to the determinism idle step, so it is matched as [0-9S].
  sclk=$(grep -o 'sclk clock level: [0-9S]*: ([0-9]*Mhz)' "$f" \
         | grep -o '([0-9]*Mhz)' | tr -d '()' | sort -u | tr '\n' ',' \
         | sed 's/,$//')
  lvl=$(grep -o 'Performance Level: [a-z_]*' "$f" | awk '{print $3}' | sort -u \
        | tr '\n' ',' | sed 's/,$//')
  echo "$stamp sclk={$sclk} perf_level=$lvl"
}

BEFORE=$(summarize "$OUT/clocks/during_run.txt")
AFTER=$(summarize "$OUT/clocks/after_run.txt")
echo "clocks before: $BEFORE"
echo "clocks after : $AFTER"

docker exec -w "$OUT" dhk-gemmrs python3 mk_ablation_json.py \
  "$LOG" "$OUT/ablation.json" \
  "iters=40" "clocks_before=$BEFORE" "clocks_after=$AFTER" \
  "arms_rebuilt=forced by tools/reattribute.sh before the run" \
  "driver=harness/exp_ablation.py via tools/reattribute.sh 1617 12"

echo
echo "===== validation: does ablation.json parse, and does it have 6 shapes? ====="
docker exec -w "$OUT" dhk-gemmrs python3 -c \
  "import json; d=json.load(open('ablation.json')); print('shapes:', len(d['shapes'])); print('gate:', d['freshness_gate'])"
