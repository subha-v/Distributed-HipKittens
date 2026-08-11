#!/usr/bin/env bash
# LF-normalized hashes of the node copies, plus a direct test for whether the
# node's hk_submission.py already carries the collective-vote fix.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
KS=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "===== node LF-normalized sha256 ====="
for f in $KS/gemm_rs_mi300x.cpp $KS/gemm_rs_mi300x_hk_adapter.cuh \
         $KS/gemm_rs_mi300x_constants.cuh $KS/gemm_rs_mi300x_host_abi.hpp \
         $ON/harness/hk_submission.py $ON/tools/run_ours_evaluator.sh; do
  printf '%s  %s\n' "$(tr -d '\r' < $f | sha256sum | cut -c1-64)" "$f"
done

echo
echo "===== does the node hk_submission.py have the vote fix? ====="
grep -c 'cache vote' $ON/harness/hk_submission.py 2>&1
grep -n 'all_gather_object(votes' $ON/harness/hk_submission.py 2>&1 || echo "NO VOTE FIX ON NODE"
wc -l $ON/harness/hk_submission.py

echo
echo "===== NUM_REDUCER / notable kernel knobs on node (read-only) ====="
grep -n 'NUM_REDUCER_CTAS' $KS/gemm_rs_mi300x_constants.cuh 2>&1 | head -8

echo
echo "===== DONE ====="
