#!/usr/bin/env bash
# exp_07 step 1: read the evaluator's clear_l2_cache / timed region, and check
# the node is clean and clocks are pinned. Read-only; no GPU work.
set -u

EV=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== A. node processes (rocm-smi --showpids) ====="
rocm-smi --showpids 2>&1 | head -40

echo
echo "===== B. clocks / perf determinism ====="
rocm-smi --showclocks 2>&1 | grep -Ei 'sclk|mclk|GPU\[0\]|GPU\[7\]' | head -20
echo "--- perf level ---"
rocm-smi --showperflevel 2>&1 | head -20

echo
echo "===== C. evaluator utils.py: clear_l2_cache ====="
if [ -f "$EV/utils.py" ]; then
  grep -n -A 25 'def clear_l2_cache' "$EV/utils.py"
else
  echo "MISSING: $EV/utils.py"
  ls -la "$EV" 2>&1 | head -30
fi

echo
echo "===== D. evaluator utils.py: _clone_data ====="
grep -n -B 3 -A 20 'def _clone_data' "$EV/utils.py" 2>&1

echo
echo "===== E. eval.py timed region (grep clear_l2_cache call sites) ====="
grep -rn 'clear_l2_cache' "$EV"/*.py 2>&1

echo
echo "===== F. the timed loop itself ====="
grep -n -B 10 -A 40 'def _run_distributed_benchmark' "$EV/eval.py" 2>&1 | head -90

echo
echo "===== G. our harness tree present? ====="
ls -la "$ON/harness"/*.py 2>&1 | head -30
echo "--- build dir ---"
ls -la "$ON/harness/build" 2>&1 | head -20

echo
echo "===== H. containers ====="
docker ps --format '{{.Names}}\t{{.Status}}' 2>&1 | head -20
echo "DONE-PROBE"
