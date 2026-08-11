#!/usr/bin/env bash
# exp_07 step 4: the graded protocol on ALL SIX shapes, one process pool, so the
# table is directly comparable to the evaluator column. Plus a look at what the
# original evaluator run actually recorded per iteration.
set -u

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_07_cold_l2
TAG=${1:-all6}
ITERS=${2:-25}
REPS=${3:-2}

echo "===== node clean check ====="
if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"; rocm-smi --showpids 2>&1 | head -30; exit 3
fi
echo "clean."

echo "===== run all six ====="
setsid timeout --signal=TERM 1500 \
  docker exec -w "$EXP" dhk-gemmrs \
  python3 mp_graded.py 1,2,3,4,5,6 "$ITERS" "$REPS" "logs/graded_${TAG}" \
  2>&1 | tee "$EXP/logs/graded_${TAG}.txt"
rc=${PIPESTATUS[0]}
echo "exit=$rc"

echo "===== post-run leak check ====="
sleep 3
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -12

echo
echo "===== what the ORIGINAL evaluator run recorded ====="
for d in "$ON/experiments/exp_01_evaluator_integration" "$ON/compbench"; do
  [ -d "$d" ] && echo "--- $d ---" && ls -la "$d" 2>&1 | head -40
done
echo "--- benchmark result lines in any exp_01 log ---"
grep -rlE 'benchmark\.[0-9]+\.(mean|best)' "$ON/experiments" 2>/dev/null | head -10
grep -rhoE 'benchmark\.[0-9]+\.(mean|best|std|err|worst)[^ ]* [0-9.e+-]+' \
  "$ON/experiments" 2>/dev/null | head -60

echo
echo "===== other tenants on the node right now ====="
docker ps --format '{{.Names}}\t{{.Status}}' 2>&1 | head -10
echo "DONE-$TAG"
exit $rc
