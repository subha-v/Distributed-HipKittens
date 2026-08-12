#!/usr/bin/env bash
# Capture the two reports as text artifacts so the numbers live on disk and not
# only in a terminal scrollback, then list what should be pulled back to the
# Windows worktree for the commit.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
python3 "$D/report.py"       > "$D/remeasure_report_ladder.txt" 2>&1
python3 "$D/compare_runs.py" > "$D/remeasure_report_delta.txt"  2>&1
for f in "$D"/raw/ladder/lad_s*.rank0.json; do
  python3 "$D/show_orders.py" "$f"
done > "$D/remeasure_arm_orders.txt" 2>&1
echo "artifacts:"
for f in ladders.json geometry.json remeasure_delta.json \
         remeasure_report_ladder.txt remeasure_report_delta.txt \
         remeasure_arm_orders.txt; do
  [ -f "$D/$f" ] && echo "  $f  $(stat -c%s "$D/$f") B" || echo "  MISSING $f"
done
echo "module sha under test: $(cat "$D/logs/module_sha_under_test.txt" 2>/dev/null)"
