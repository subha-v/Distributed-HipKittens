#!/usr/bin/env bash
# exp_01 step 1: the PROVEN multi-process path must still work. This guards
# against a regression in the vote fix itself before the evaluator gets blamed.
# HK_DEBUG=1 deliberately: this step is diagnostic, never a timing measurement.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/experiments/exp_01_evaluator_integration/logs
mkdir -p $OUT

echo "===== build .so fingerprints (detect a concurrent rebuild later) ====="
sha256sum $ON/harness/build/gemm_rs_mi300x.so $ON/harness/build/dhk_rt.so 2>&1
ls -la --time-style=full-iso $ON/harness/build/gemm_rs_mi300x.so 2>&1

echo
echo "===== mp_smoke, 8 ranks, HK_DEBUG=1 ====="
docker exec -w $ON/harness dhk-gemmrs bash -c \
  'cp hk_submission.py submission.py; HK_DEBUG=1 timeout 600 python3 -u mp_smoke.py' \
  > $OUT/mp_smoke.log 2>&1
rc=$?
echo "exit=$rc"

echo
echo "===== verdict lines ====="
grep -E 'allclose|max\|diff\||ERROR BITS|Traceback|FAILED|Error|error' $OUT/mp_smoke.log | head -30

echo
echo "===== last 40 lines ====="
tail -40 $OUT/mp_smoke.log

echo
echo "===== how many ranks reached each step ====="
for step in 'enter custom_kernel' 'cache vote' 'state ready' 'setup barrier returned' \
            'device synchronize returned' 'launch issued' 'launch synchronized'; do
  printf '%-32s %s\n' "$step" "$(grep -c "$step" $OUT/mp_smoke.log)"
done

echo
echo "===== STEP1 DONE rc=$rc ====="
exit $rc
