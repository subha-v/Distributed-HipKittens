#!/usr/bin/env bash
# Our protocol pairs rank A's epoch e with rank B's epoch e. If the evaluator's
# per-rank benchmark loop chooses its iteration count from LOCALLY measured
# time, ranks can diverge in call count, epochs desync, and every reducer times
# out. Establish exactly how the loop terminates and where the barriers are.
set -uo pipefail
# Read the pristine copy: the staged arms are written by a root container.
CWD=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd

echo "===== _run_distributed_benchmark (per rank) ====="
sed -n '250,320p' "$CWD/eval.py"

echo
echo "===== the distributed benchmark body ====="
sed -n '311,412p' "$CWD/eval.py"

echo
echo "===== run_multi_gpu_benchmark (the pool side) ====="
sed -n '413,470p' "$CWD/eval.py"

echo
echo "===== every barrier / all_reduce in eval.py and utils.py ====="
grep -nE 'barrier|all_reduce|broadcast|dist\.' "$CWD/eval.py" "$CWD/utils.py" | head -30

echo
echo "===== how custom_kernel is invoked, and what wraps it ====="
grep -n -B6 -A20 'custom_kernel' "$CWD/eval.py" "$CWD/utils.py" | head -60
