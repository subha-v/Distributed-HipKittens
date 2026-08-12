#!/usr/bin/env bash
# exp_10 probe 12 -- NO GPU (grep only).
# rank-1's cached fast path does bias.data_ptr() unconditionally, which is a
# TypeError for the three graded shapes with has_bias=False. Yet eval.py ran
# shape 1 for 100 iterations. Find out what eval.py does that we do not --
# most likely its _clone_data materialises a bias.
set -uo pipefail

ARM=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1

echo "===== A. eval.py _clone_data (127..155) ====="
awk 'NR>=125 && NR<=156 {printf "%d: %s\n", NR, $0}' "$ARM/eval.py"

echo
echo "===== B. eval.py benchmark loop: is there a try/except swallowing errors? ====="
awk 'NR>=336 && NR<=412 {printf "%d: %s\n", NR, $0}' "$ARM/eval.py"

echo
echo "===== C. rank-1 origin() -- the torch fallback ====="
awk 'NR>=1704 && NR<=1742 {printf "%d: %s\n", NR, $0}' "$ARM/submission.py"

echo
echo "===== D. every bias reference in rank-1 ====="
grep -nE 'bias' "$ARM/submission.py" | grep -vE '^\s*#' | head -40
echo "===== DONE p12 ====="
