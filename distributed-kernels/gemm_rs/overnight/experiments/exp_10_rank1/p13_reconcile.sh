#!/usr/bin/env bash
# exp_10 probe 13 -- NO GPU (grep only).
# CONTRADICTION to resolve: rank-1's cached fast path dereferences bias
# unconditionally, and eval.py's _clone_data passes None through, yet eval.py
# reported 100 clean runs on shape 1 (has_bias=False). One of those is wrong,
# and the answer decides whether the evaluator numbers are trustworthy.
set -uo pipefail

ARM=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1
SO=$ARM/benchmark.stdout.txt
SE=$ARM/benchmark.stderr.txt

echo "===== A. how many times did each shape COMPILE its kernel? ====="
echo "(one line per compile; 1 per shape = cache sticks, ~100 = recompiled every call)"
for pat in '(64, 7168, 2304)' '(512, 4096, 1536)' '(2048, 2880, 360)'; do
  printf '  %-22s %s\n' "$pat" "$(grep -cF "$pat" "$SO" 2>/dev/null)"
done

echo
echo "===== B. AttributeError / NoneType anywhere in the eval run? ====="
grep -cE "AttributeError|NoneType" "$SE" "$SO" 2>/dev/null
grep -nE "AttributeError|NoneType" "$SE" 2>/dev/null | head -5

echo
echo "===== C. how many 'start clear' (destroy_process_group) events? ====="
echo "  start clear count: $(grep -oc 'start clear' "$SO" 2>/dev/null || grep -o 'start clear' "$SO" 2>/dev/null | wc -l)"
echo "  init here count  : $(grep -o 'init here' "$SE" 2>/dev/null | wc -l)"
echo "  init cost count  : $(grep -o 'init cost' "$SO" 2>/dev/null | wc -l)"

echo
echo "===== D. is pre_compile_cache set for the no-bias path? read it verbatim ====="
awk 'NR>=1440 && NR<=1448 {printf "%d: %s\n", NR, $0}' "$ARM/submission.py"
echo "  ..."
awk 'NR>=1530 && NR<=1548 {printf "%d: %s\n", NR, $0}' "$ARM/submission.py"

echo
echo "===== E. eval.py: is recheck True for benchmark mode? ====="
grep -nE 'recheck|_run_distributed_benchmark|run_multi_gpu_benchmark|run_benchmarking' "$ARM/eval.py" | head -20
awk 'NR>=469 && NR<=502 {printf "%d: %s\n", NR, $0}' "$ARM/eval.py"

echo
echo "===== F. the first 40 lines of the eval stdout for shape 1 ====="
head -40 "$SO" 2>/dev/null
echo "===== DONE p13 ====="
