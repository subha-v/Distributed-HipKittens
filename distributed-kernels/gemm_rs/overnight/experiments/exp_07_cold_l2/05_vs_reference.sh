#!/usr/bin/env bash
# exp_07 step 5: ours vs the reference GEMM+RCCL through the identical graded
# protocol, same process pool, interleaved. The only same-run denominator we
# have; the evaluator's own reference column was measured in a different run.
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_07_cold_l2
TAG=${1:-vs}
ITERS=${2:-25}
REPS=${3:-2}

mkdir -p "$EXP/logs"
sed -i 's/\r$//' "$EXP/mp_graded.py"

echo "===== node clean check ====="
if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"; rocm-smi --showpids 2>&1 | head -30; exit 3
fi
echo "clean."

docker exec -w "$EXP" dhk-gemmrs python3 -m py_compile mp_graded.py || exit 4

setsid timeout --signal=TERM 1500 \
  docker exec -w "$EXP" dhk-gemmrs \
  python3 mp_graded.py 1,2,3,4,5,6 "$ITERS" "$REPS" "logs/graded_${TAG}" vs \
  2>&1 | tee "$EXP/logs/graded_${TAG}.txt"
rc=${PIPESTATUS[0]}
echo "exit=$rc"

sleep 3
echo "===== post-run leak check ====="
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -12
echo "DONE-$TAG"
exit $rc
