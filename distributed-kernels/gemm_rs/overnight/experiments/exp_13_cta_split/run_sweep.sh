#!/usr/bin/env bash
# exp_13 driver: run the NR sweep in dhk-gemmrs after the node has drained.
#
#   run_sweep.sh [arms] [passes] [shapes] [iters_scale]
#   e.g. run_sweep.sh 4,8,16,24,32,40,48,56,64,80 0 all 1
#
# Waits for other KFD processes rather than aborting (we hold the lease, but a
# straggler of ours must be allowed to finish or be reaped deliberately).
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_13_cta_split
L=$D/logs
mkdir -p "$L"
STAMP=$(date +%H%M%S)

for i in $(seq 1 90); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  [ "$n" = "0" ] && break
  echo "waiting for $n KFD pid(s) to drain (attempt $i/90)"
  sleep 10
done
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
if [ "$n" != "0" ]; then echo "NODE STILL DIRTY ($n pids) - refusing to time"; exit 1; fi
echo "node clean"

echo "=== clocks (must be pinned ~1900 for any timing) ==="
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk' | head -8
rocm-smi 2>&1 | grep -iE 'perf|DPM' | head -4

echo "=== build state (this sweep must NOT rebuild: runtime override only) ==="
ls -l $ON/harness/build/*.so 2>/dev/null
md5sum $ON/harness/build/gemm_rs_mi300x.so $ON/harness/build/dhk_rt.so 2>/dev/null

echo "=== sweep: $* ==="
docker exec -w $ON/harness dhk-gemmrs timeout 7200 \
  python3 -u $D/sweep.py "$@" 2>&1 | tee "$L/sweep_$STAMP.log"
rc=${PIPESTATUS[0]}
echo "sweep exit $rc"
ls -l $D/*.json 2>/dev/null
exit $rc
