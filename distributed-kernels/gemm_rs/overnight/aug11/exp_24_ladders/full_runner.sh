#!/usr/bin/env bash
# exp_24 full-ladder runner. Detached: it must survive the ssh connection that
# launched it, because the lease wait alone can be an hour and the ladder itself
# several more.
#
# Lease discipline: acquire before anything touches a GPU, and release on EVERY
# exit path including failure and SIGTERM. gpu_lease.sh performs its own
# wait-for-drain, so run_ladders.sh's foreign-tenant abort is suppressed via
# LAD_LEASED=1 (it still pins clocks and still reports node state). Nothing here
# SIGKILLs anything: leaked HIP IPC mappings wedge this node.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
LOG=$D/logs/full_run.log
mkdir -p "$D/logs"
exec >>"$LOG" 2>&1

HELD=0
cleanup() {
  local rc=$?
  echo
  echo "=== TRAP rc=$rc at $(date -Is) ==="
  if [ "$HELD" = "1" ]; then
    bash "$ON/tools/gpu_lease.sh" release exp_24
  else
    echo "lease was never acquired by us; not releasing"
  fi
  echo "=== full runner EXIT rc=$rc at $(date -Is) ==="
}
trap cleanup EXIT
trap 'echo "SIGTERM received"; exit 143' TERM
trap 'echo "SIGINT received"; exit 130' INT

echo
echo "################################################################"
echo "=== exp_24 FULL LADDER runner start $(date -Is) pid=$$ ==="
echo "################################################################"
echo "queue position: third (exp_23 then exp_21 ahead). A long wait is expected."
bash "$ON/tools/gpu_lease.sh" status

echo
echo "=== acquiring GPU lease (wait up to 10800s) $(date -Is) ==="
if bash "$ON/tools/gpu_lease.sh" acquire exp_24 10800; then
  HELD=1
else
  echo "LEASE NOT ACQUIRED -- aborting without touching a GPU."
  bash "$ON/tools/gpu_lease.sh" status
  exit 2
fi

echo
echo "=== lease held; starting the ladder $(date -Is) ==="
export LAD_LEASED=1
export LAD_ROTATIONS="${LAD_ROTATIONS:-2}"
# 7 h cap on the ladder itself. Instrument A is allowed to overrun its estimate
# rather than truncate -- a truncated rank-1 pass is exactly the artifact the
# parser refuses -- so this cap exists only to guarantee the lease comes back.
timeout --signal=TERM 25200 bash "$D/run_ladders.sh" all
rc=$?
echo
echo "=== run_ladders.sh returned rc=$rc at $(date -Is) ==="
[ $rc -eq 124 ] && echo "NOTE: rc=124 means the 7 h cap fired; the run was TRUNCATED."
exit $rc
