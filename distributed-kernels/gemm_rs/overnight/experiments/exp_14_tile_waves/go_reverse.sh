#!/usr/bin/env bash
# Five draws with the arm CONSTRUCTION order reversed, then re-pool all draws.
#
# Purpose: the T-vs-T* contrast came out negative in 10 of 10 forward draws on
# shape 1 (median -0.91%), which is not what symmetric per-allocation noise looks
# like. If that is an allocation-ORDER effect, reversing construction must flip
# the sign; if it is something about the arms themselves, it will not. Either
# way the pooled null becomes symmetric, which is what the disjointness test
# needs to be meaningful.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
mkdir -p "$D/logs"

for attempt in $(seq 1 40); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  echo "  M0 attempt $attempt: KFD pids = $n"
  [ "$n" = "0" ] && break
  sleep 15
done
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || { echo "NODE BUSY -- not launching"; exit 1; }

for r in 1 2 3 4 5; do
  echo
  echo "########## reversed draw $r of 5 ##########"
  docker exec -e SWEEP_REVERSE=1 -w $D dhk-gemmrs timeout 3600 \
    python3 -u sweep.py 1,2,3 8 1 2>&1 | \
    tee "$D/logs/sweep_rev${r}_$(date +%H%M%S).log"
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "REVERSED DRAW $r FAILED"; exit 1; }
done

docker exec -w $D dhk-gemmrs python3 -u pool.py 2>&1 | tee "$D/logs/pool.log"
echo "POOL rc=${PIPESTATUS[0]}"
