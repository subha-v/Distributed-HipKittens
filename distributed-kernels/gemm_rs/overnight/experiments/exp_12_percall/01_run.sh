#!/usr/bin/env bash
# exp_12 step 1: decompose the graded per-call cost.
#   args: <shapes_csv> <arms_csv> <iters> <reps> <tag> <port> <timeout_s>
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_12_percall
SHAPES=${1:-1,4}
ARMS=${2:-all}
ITERS=${3:-20}
REPS=${4:-2}
TAG=${5:-a}
PORT=${6:-12610}
TMO=${7:-1200}

sed -i 's/\r$//' "$EXP"/*.py
mkdir -p "$EXP/logs"
LOG=$EXP/logs/run_${TAG}.txt

echo "===== node clean check ====="
if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"
  rocm-smi --showpids 2>&1 | head -20
  exit 3
fi
echo "clean."

docker exec -w "$EXP" dhk-gemmrs python3 -m py_compile mp_percall.py || exit 4

echo "===== run shapes=$SHAPES arms=$ARMS iters=$ITERS reps=$REPS ====="
setsid timeout --signal=TERM "$TMO" \
  docker exec -w "$EXP" dhk-gemmrs \
  python3 -u mp_percall.py "$SHAPES" "$ARMS" "$ITERS" "$REPS" "$TAG" "$PORT" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "exit=$rc"

echo "===== post-run leak check ====="
sleep 3
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -8
pgrep -af 'mp_percall' | head -5
echo "DONE-$TAG"
exit $rc
