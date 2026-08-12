#!/usr/bin/env bash
# exp_24 re-measure: the common-mode diagnostic and the within-run ratio deltas.
# Read-only, CPU only. Also quarantines the instrument-B captures, which were
# NOT re-run in this dispatch and therefore belong to the previous binary.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=============== quarantine the stale instrument-B captures ==============="
# raw/eval still holds the PREVIOUS run's evaluator output (79599cce). The
# aggregator happily folded it into the new ladders.json as `evaluator_crosscheck`,
# where it would read as this run's cross-check. It is archived under
# prev_79599cce/raw_eval; move it out of the aggregator's reach and re-aggregate
# so instrument B is absent rather than mislabelled.
if [ -d "$D/raw/eval" ]; then
  mv "$D/raw/eval" "$D/raw/eval_prev_79599cce_DO_NOT_PARSE"
  echo "  moved raw/eval -> raw/eval_prev_79599cce_DO_NOT_PARSE"
  mkdir -p "$D/raw/eval"
fi
python3 "$D/ladders.py" --root "$D" --out "$D/ladders.json" --ladder-dir ladder \
  --expect-a 6 --expect-b 0 | tail -22
echo "  re-aggregate rc=$?"

echo
echo "=============== the diagnostic ==============="
python3 "$D/compare_runs.py"
