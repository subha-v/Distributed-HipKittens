#!/usr/bin/env bash
# exp_07 step 3: graded protocol in 8 processes, arms full/noclone/noflush/bare.
#   args: <shapes_csv> <iters> <reps> <tag>
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_07_cold_l2
SHAPES=${1:-4,5}
ITERS=${2:-25}
REPS=${3:-2}
TAG=${4:-a}

mkdir -p "$EXP/logs"
sed -i 's/\r$//' "$EXP/mp_graded.py"
LOG=$EXP/logs/graded_${TAG}.txt

echo "===== what _clone_data actually is, in the evaluator ====="
EV=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
grep -rn -A 12 'def _clone_data' "$EV"/*.py 2>&1 | head -30
echo "----- the timed region, eval.py 336-372 -----"
sed -n '336,372p' "$EV/eval.py" 2>&1

echo "===== node clean check ====="
if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"
  rocm-smi --showpids 2>&1 | head -30
  exit 3
fi
echo "clean."

echo "===== syntax check ====="
docker exec -w "$EXP" dhk-gemmrs python3 -m py_compile mp_graded.py || exit 4

echo "===== run: shapes=$SHAPES iters=$ITERS reps=$REPS ====="
setsid timeout --signal=TERM 1500 \
  docker exec -w "$EXP" dhk-gemmrs \
  python3 mp_graded.py "$SHAPES" "$ITERS" "$REPS" "logs/graded_${TAG}" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "exit=$rc"

echo "===== post-run leak check ====="
sleep 3
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -12
pgrep -af 'mp_graded' | head -10
echo "DONE-$TAG"
exit $rc
