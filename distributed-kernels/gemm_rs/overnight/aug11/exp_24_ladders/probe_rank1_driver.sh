#!/usr/bin/env bash
# READ-ONLY: r1_eval.sh is the script that actually got rank-1 through the
# official evaluator in exp_10 (result.md section 4). run_rank1_bench3.sh looks
# like it predates repair #6, so establish which one carries the buffer-ops knob
# before the orchestrator picks a driver.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
echo "########## experiments/exp_10_rank1/r1_eval.sh ##########"
cat "$ON/experiments/exp_10_rank1/r1_eval.sh" 2>&1
echo
echo "########## experiments/exp_10_rank1/p4_stage.sh ##########"
cat "$ON/experiments/exp_10_rank1/p4_stage.sh" 2>&1
echo
echo "########## which drivers set AMDGCN_USE_BUFFER_OPS ##########"
grep -rln 'AMDGCN_USE_BUFFER_OPS' "$ON/tools" "$ON/experiments" "$ON/aug11" 2>/dev/null
echo
echo "########## tools/compat/sitecustomize.py head ##########"
sed -n '1,30p' "$ON/tools/compat/sitecustomize.py" 2>&1
echo
echo "########## compbench/ contents (what is already staged) ##########"
ls -la "$ON/compbench/" 2>&1
for a in ours reference rank1; do echo "--- $a ---"; ls "$ON/compbench/$a" 2>&1 | head -20; done
echo "########## DONE ##########"
