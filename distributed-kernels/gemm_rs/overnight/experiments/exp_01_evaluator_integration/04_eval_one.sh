#!/usr/bin/env bash
# exp_01 step 2: the official evaluator, ONE graded shape, HK_DEBUG=1.
# Diagnostic only -- HK_DEBUG=1 puts a synchronize inside every timed call, so
# no number out of this run is reportable.
#
# The discriminator: HK_ONE=1 is a single test case, hence a single process
# group. mp_smoke hung at its SECOND init_process_group, so if one case passes
# here the fault needs >=2 groups.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$ON/compbench/ours
OUT=$ON/experiments/exp_01_evaluator_integration/logs
mkdir -p $OUT

HK_ONE=1 HK_DEBUG=1 timeout 900 bash $ON/tools/run_ours_evaluator.sh \
  > $OUT/eval_one.log 2>&1
rc=$?
echo "run_ours_evaluator rc=$rc"

echo
echo "===== popcorn / result lines ====="
grep -E 'benchmark|check|pass|fail|mean|error|Error|Traceback' $OUT/eval_one.log | head -40

echo
echo "===== per-step rank counts in evaluator stderr ====="
S=$DIR/benchmark.stderr.txt
docker exec dhk-gemmrs bash -c "test -f $S && wc -l $S" 2>&1
for step in 'enter custom_kernel' 'cache vote' 'cache hit' 'cache miss' \
            'get_ipc_handle done' 'all_gather_object done' 'opened all peers' \
            'descriptors uploaded' 'setup barrier$' 'setup barrier returned' \
            'device synchronize returned' 'state ready' 'launch issued' \
            'launch synchronized' 'ERROR BITS' 'FAILED peer'; do
  n=$(docker exec dhk-gemmrs bash -c "grep -c '$step' $S 2>/dev/null" 2>/dev/null)
  printf '%-32s %s\n' "$step" "${n:-0}"
done

echo
echo "===== every cache vote line ====="
docker exec dhk-gemmrs bash -c "grep -h 'cache vote' $S 2>/dev/null | head -30" 2>&1

echo
echo "===== IPC failures, if any ====="
docker exec dhk-gemmrs bash -c "grep -h 'FAILED peer\|AlreadyMapped\|hipIpc' $S 2>/dev/null | head -20" 2>&1

echo
echo "===== tail of the evaluator log ====="
tail -60 $OUT/eval_one.log

echo
echo "===== STEP2 DONE rc=$rc ====="
