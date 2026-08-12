#!/usr/bin/env bash
#   args: <shape_index_0based> <iters> <port>
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_12_percall
IDX=${1:-0}
ITERS=${2:-3000}
PORT=${3:-12620}

sed -i 's/\r$//' "$EXP"/*.py
mkdir -p "$EXP/logs"

if ! rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs currently running'; then
  echo "REFUSING: GPU processes present"; rocm-smi --showpids 2>&1 | head -20
  exit 3
fi
docker exec -w "$EXP" dhk-gemmrs python3 -m py_compile 03_hostprofile.py || exit 4

setsid timeout --signal=TERM 900 \
  docker exec -w "$EXP" dhk-gemmrs \
  python3 -u 03_hostprofile.py "$IDX" "$ITERS" "$PORT" \
  2>&1 | tee "$EXP/logs/hostprof_s$((IDX+1)).txt"
rc=${PIPESTATUS[0]}
sleep 2
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -5
echo "exit=$rc"
echo "DONE-hostprof"
exit $rc
