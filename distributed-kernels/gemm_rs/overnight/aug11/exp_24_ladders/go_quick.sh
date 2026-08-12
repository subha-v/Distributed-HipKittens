#!/usr/bin/env bash
# exp_24 LAD_QUICK launcher: re-verify the node is clean IMMEDIATELY before the
# launch, abort if it is not, then run the ~6 min end-to-end validation under
# setsid + timeout. SIGTERM only -- never SIGKILL a GPU process on this node,
# leaked HIP IPC wedges it.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=== node check immediately before launch $(date -Is) ==="
rocm-smi --showpids 2>&1 | sed -n '3,20p'
pids=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/{print $1}')
fds=$(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)
echo "kfd pids: [$pids]   kfd fds: $fds"
if [ -n "$pids" ] || [ "$fds" != "0" ]; then
  echo "ABORT: node is not clean. Not launching."
  exit 1
fi
echo "node CLEAN -- launching"

echo
echo "=== syntax re-check after the two fixes ==="
bash -n "$D/run_ladders.sh" && echo "OK run_ladders.sh"
docker exec dhk-gemmrs bash -lc "cd $D && python3 -m py_compile ladders.py ladder_mp.py && echo 'OK python'" 2>&1 | tail -2

echo
echo "=== LAD_QUICK=1 run_ladders.sh all ==="
cd "$D" || exit 1
LAD_QUICK=1 setsid timeout --signal=TERM 1500 bash "$D/run_ladders.sh" all 2>&1
rc=$?
echo "=== quick run rc=$rc ==="

echo
echo "=== post-run node state (must be clean again) ==="
rocm-smi --showpids 2>&1 | sed -n '3,12p'
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
exit $rc
