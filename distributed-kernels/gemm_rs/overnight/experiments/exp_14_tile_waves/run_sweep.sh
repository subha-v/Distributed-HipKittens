#!/usr/bin/env bash
# E4b screening sweep, repeated in INDEPENDENT PROCESSES.
#
# Why repeats and not just more passes. The documented bias on this node is
# PER-ALLOCATION: two arms with identical code and identical operands separated
# by 4.28% on shape 6 while the positional residual stayed inside 0.47%. Extra
# passes inside one process re-measure the SAME allocation draw and cannot see
# that bias at all -- they only tighten the within-draw jitter. Only a fresh
# process draws fresh hipMalloc returns. So: `repeats` independent draws, each
# with a complete positional rotation, pooled afterwards by pool.py, which
# requires DISJOINT RANGES across draws before it will call anything a win.
#
# Node discipline: wait for other GPU jobs to drain rather than aborting or
# killing them; SIGKILL on a HIP-IPC process can wedge the node.
#
#   run_sweep.sh [shapes] [passes] [scale] [repeats]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
SHAPES=${1:-1,2,3}
PASSES=${2:-8}
SCALE=${3:-1}
REPEATS=${4:-5}
mkdir -p "$D/logs"

echo "########## M0 node state ##########"
for attempt in $(seq 1 40); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  echo "  attempt $attempt: KFD pids = $n"
  [ "$n" = "0" ] && break
  rocm-smi --showpids 2>/dev/null | head -12
  sleep 15
done
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
if [ "$n" != "0" ]; then
  echo "NODE STILL BUSY after 10 minutes of waiting -- not launching"
  exit 1
fi

echo "########## clocks ##########"
bash $ON/tools/set_clocks.sh pin 1900 2>&1 | tail -4

echo "########## freshness: the sweep module must postdate the source ##########"
ls -l --time-style=+%s $D/../../../gemm_rs_mi300x.cpp \
      $ON/harness/build/gemm_rs_mi300x_tilesweep.so 2>&1 | awk '{print "  "$0}'
echo "  (push.ps1 resets source mtimes, so this is informational only --"
echo "   the real freshness guarantee is that build_sweep.sh rm's the .so)"

rc=0
for r in $(seq 1 "$REPEATS"); do
  echo
  echo "########## draw $r of $REPEATS: shapes=$SHAPES passes=$PASSES scale=$SCALE ##########"
  docker exec -w $D dhk-gemmrs timeout 3600 \
    python3 -u sweep.py "$SHAPES" "$PASSES" "$SCALE" 2>&1 | \
    tee "$D/logs/sweep_draw${r}_$(date +%H%M%S).log"
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "DRAW $r FAILED"; rc=1; break; }
done
echo "SWEEP DONE rc=$rc"
exit $rc
