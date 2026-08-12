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
#
# EVERY shared tool is snapshotted first, NORMALIZED, syntax-checked, and only the
# snapshot is executed.
#
# Why, measured tonight: at 05:30 this run acquired the lease and died 0 s later
# with `gpu_lease.sh: 24: set: Illegal option -o pipefail` followed by
# `exit: 2: numeric argument required`. The cause is not a race -- it is that
# tools/gpu_lease.sh arrived with **Windows CRLF line endings**, so bash read
# `pipefail\r` as an option name and `2\r` as an exit status. This is the exact
# trap HANDOFF.md documents, and it is silent in `cat` and invisible to grep.
# The lease was acquired and then instantly lost, twice, before the syntax check
# below named it in one line.
#
# So: strip CR, then refuse to run anything that does not parse. Normalizing a
# private copy is behaviour-preserving (a carriage return is not syntax in any of
# these files) and it does not touch a tool this experiment does not own -- the
# CRLF defect is reported upward instead. The snapshot doubles as exact
# provenance: the sha256s in provenance.txt are of the bytes that actually ran.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
SNAP=$D/toolsnap
LOG=$D/logs/full_run.log
mkdir -p "$D/logs"
exec >>"$LOG" 2>&1

echo
echo "################################################################"
echo "=== exp_24 FULL LADDER runner start $(date -Is) pid=$$ ==="
echo "################################################################"

echo "=== snapshotting shared tools -> $SNAP ==="
rm -rf "$SNAP"; mkdir -p "$SNAP"
NEED="gpu_lease.sh set_clocks.sh patch_rank1.py run_ours_evaluator.sh
      run_reference_arm.sh run_rank1_bench3.sh"
for f in $NEED; do
  [ -f "$ON/tools/$f" ] || { echo "FATAL: $ON/tools/$f missing"; exit 3; }
  crlf=$(grep -c $'\r' "$ON/tools/$f" 2>/dev/null || echo 0)
  tr -d '\r' < "$ON/tools/$f" > "$SNAP/$f" || { echo "FATAL: cannot snapshot $f"; exit 3; }
  # Syntax-check the normalized snapshot. This is the one moment a truncated or
  # CRLF-damaged tool can be caught for free rather than at hour three.
  if [ "${f##*.}" = "sh" ]; then
    bash -n "$SNAP/$f" || { echo "FATAL: $f is not valid bash even after CR strip"; exit 3; }
  else
    python3 -m py_compile "$SNAP/$f" || { echo "FATAL: $f is not valid python"; exit 3; }
  fi
  tag="lf"
  [ "$crlf" != "0" ] && tag="CRLF x$crlf -> NORMALIZED (report upward)"
  echo "  OK $f  sha256 $(sha256sum "$SNAP/$f" | cut -c1-16)  $tag"
done
LEASE=$SNAP/gpu_lease.sh
export LAD_TOOLSNAP=$SNAP

HELD=0
cleanup() {
  local rc=$?
  echo
  echo "=== TRAP rc=$rc at $(date -Is) ==="
  if [ "$HELD" = "1" ]; then
    bash "$LEASE" release exp_24
  else
    echo "lease was never acquired by us; not releasing"
  fi
  echo "=== full runner EXIT rc=$rc at $(date -Is) ==="
}
trap cleanup EXIT
trap 'echo "SIGTERM received"; exit 143' TERM
trap 'echo "SIGINT received"; exit 130' INT

echo
echo "queue position: behind whoever holds it now. A long wait is expected."
bash "$LEASE" status

# Retry a non-TIMEOUT acquire failure. rc=2 is the tool's "someone else still
# holds it at my deadline", which is final; anything else (rc=1 = node dirty at
# the deadline, rc>=3 = the tool itself blew up) is worth one more attempt,
# because tonight's two failures were both transient environment damage rather
# than a real queue.
for attempt in 1 2 3; do
  echo
  echo "=== acquiring GPU lease, attempt $attempt (wait up to 10800s) $(date -Is) ==="
  bash "$LEASE" acquire exp_24 10800
  arc=$?
  if [ $arc -eq 0 ]; then HELD=1; break; fi
  echo "acquire attempt $attempt failed rc=$arc"
  if [ $arc -eq 2 ]; then
    echo "rc=2 is a genuine queue timeout after 3 h -- not retrying."
    bash "$LEASE" status
    exit 2
  fi
  bash "$LEASE" status
  sleep 30
done
if [ "$HELD" != "1" ]; then
  echo "LEASE NOT ACQUIRED after 3 attempts -- aborting without touching a GPU."
  exit 4
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
