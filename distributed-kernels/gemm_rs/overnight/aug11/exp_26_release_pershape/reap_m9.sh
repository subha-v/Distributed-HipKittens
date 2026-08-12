#!/usr/bin/env bash
# Graceful reap of the faulted m9 process, and a read of the lease protocol this
# experiment should have been using.
#
# SIGTERM only. This kernel uses HIP IPC and SIGKILL on a GPU process can leak
# IPC mappings and wedge the node for every tenant, which is a far worse outcome
# than a process that takes a minute to leave.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "########## the lease tool (read-only) ##########"
if [ -f "$ON/tools/gpu_lease.sh" ]; then
  sed -n '1,60p' "$ON/tools/gpu_lease.sh"
  echo "-- lease state --"
  ls -la /tmp/dhk_gpu_lease* 2>/dev/null || echo "  (no /tmp lease artefacts)"
  cat /tmp/dhk_gpu_lease/holder 2>/dev/null || true
else
  echo "  (tools/gpu_lease.sh does not exist)"
fi

echo
echo "########## before ##########"
pgrep -af 'm9_stale_slot' || echo "  (already gone)"
rocm-smi --showpids 2>/dev/null | head -12

for sig in TERM TERM; do
  pids=$(pgrep -f 'm9_stale_slot' | tr '\n' ' ')
  [ -z "$pids" ] && break
  echo "sending SIG$sig to: $pids"
  kill -s $sig $pids 2>/dev/null
  sleep 20
done

echo
echo "########## after ##########"
pgrep -af 'm9_stale_slot' || echo "  m9 is gone"
echo "-- KFD pids --"
rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l
echo "-- gpus --"
rocm-smi 2>&1 | tail -12 | head -10
