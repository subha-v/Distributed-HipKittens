#!/usr/bin/env bash
# exp_07 step 2: paired warm/cold-L2 single-shot sweep.
#   args: <arms_csv_mib> <iters> <reps> <shapes_csv|all> <tag>
# e.g.  "0,256 15 3 all main"
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_07_cold_l2
ARMS=${1:-0,256}
ITERS=${2:-15}
REPS=${3:-3}
SHAPES=${4:-all}
TAG=${5:-main}

mkdir -p "$EXP/logs"
LOG=$EXP/logs/coldl2_${TAG}.txt

# The file arrives from a Windows checkout; bash/python both choke on stray \r.
sed -i 's/\r$//' "$ON/harness/m7_bench_coldl2.py"
echo "===== syntax check ====="
docker exec -w "$ON/harness" dhk-gemmrs python3 -m py_compile m7_bench_coldl2.py || exit 4
echo "compiles."

echo "===== node clean check ====="
PIDS=$(rocm-smi --showpids 2>&1)
echo "$PIDS" | grep -E 'No KFD PIDs|PID' | head -10
if ! echo "$PIDS" | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"
  echo "$PIDS" | head -30
  exit 3
fi
echo "clean."

echo "===== perf level ====="
rocm-smi --showperflevel 2>&1 | grep -c perf_determinism

echo "===== run: arms=$ARMS iters=$ITERS reps=$REPS shapes=$SHAPES tag=$TAG ====="
setsid timeout --signal=TERM 1500 \
  docker exec -w "$ON/harness" dhk-gemmrs \
  python3 m7_bench_coldl2.py "$ARMS" "$ITERS" "$REPS" "$SHAPES" "$TAG" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "exit=$rc"

echo "===== post-run leak check ====="
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -10
if [ -f "$ON/harness/coldl2_results_${TAG}.json" ]; then
  cp "$ON/harness/coldl2_results_${TAG}.json" "$EXP/logs/"
  echo "json copied"
fi
echo "DONE-$TAG"
exit $rc
