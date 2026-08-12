#!/usr/bin/env bash
# exp_21 campaign: ONE lease hold, four phases, lease released on every exit path.
#
# Runs on the HOST (rocm-smi and the lease live there), driving the container per
# phase. Launch it detached -- `acquire` blocks while another agent's campaign
# holds the node, and a blocking ssh would just time out:
#
#   setsid timeout --signal=TERM --kill-after=300 25200 \
#     bash campaign.sh >> logs/campaign_boot.log 2>&1 &
#
# Phase chain, aborting the rest on failure:
#   probe   full-size fine-grained allocation + one launch of all seven kernels +
#           one checksum. Tests the two things the ISA pass could not: the
#           65,536 B DYNAMIC LDS request launching at all, and 8 x ~4 GiB of
#           fine-grained heap. --quick shrinks c_work, so it does NOT test the
#           allocation; that is why probe exists and runs first.
#   quick   1 rotation, 3 CTA points per mode. Any anomaly here is blocking.
#   full    the figure data: ~150 points x 5 rotations.
#   coarse  payload-granularity cross-check (coarse/cached destination). If this
#           reports a HIGHER isolated mode-c plateau than fine-grained at
#           protocol=0, the issuing XCD's L2 absorbed the peer store and the
#           number is L2 bandwidth, not fabric bandwidth.
#
# Never SIGKILLs: leaked HIP IPC mappings wedge this node, so every timeout here
# is --signal=TERM with a generous --kill-after.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
OWNER=exp_21
WAIT_S=${WAIT_S:-5400}
PHASES=${PHASES:-"probe quick full coarse"}

mkdir -p "$EXP/logs"
LOG=$EXP/logs/campaign_$(date -u +%Y%m%dT%H%M%SZ).log
ln -sf "$LOG" "$EXP/logs/campaign_latest.log"
exec >>"$LOG" 2>&1

say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# Owner-checked, so this is safe even if `acquire` timed out and we never held it.
cleanup() { say "releasing lease"; bash "$ON/tools/gpu_lease.sh" release "$OWNER"; }
trap cleanup EXIT INT TERM

say "campaign start (pid $$), phases: $PHASES"
say "waiting for the GPU lease (up to ${WAIT_S}s; exp_23 is ahead in the queue)"
bash "$ON/tools/gpu_lease.sh" acquire "$OWNER" "$WAIT_S"
rc=$?
if [ "$rc" != "0" ]; then
  say "ABORT: could not acquire the lease (rc=$rc)"
  exit 2
fi

say "--- clocks before (pinned ~1900 is a precondition for every number below) ---"
bash "$ON/tools/set_clocks.sh" show 2>&1 | head -20
if ! rocm-smi --showclocks 2>/dev/null | grep -i sclk | grep -qE '1[6-9][0-9][0-9]Mhz'; then
  say "sclk does not look pinned; attempting to pin (needs root -- may refuse)"
  bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | head -20
fi

status=0
for phase in $PHASES; do
  say "########## phase: $phase ##########"
  t0=$(date -u +%s)
  docker exec dhk-gemmrs bash "$EXP/run_sweep.sh" "$phase"
  prc=$?
  say "phase $phase exit=$prc after $(( $(date -u +%s) - t0 ))s"
  if [ "$prc" != "0" ]; then
    # probe and quick are gates: a failure there means every later number would be
    # measured on a fixture we know is broken. coarse is a cross-check, so its
    # failure is a result, not a reason to discard the full sweep.
    status=$prc
    if [ "$phase" = "coarse" ]; then
      say "coarse failed; that is itself evidence (see result.md) -- continuing"
      continue
    fi
    say "ABORT: $phase is a gate; skipping the remaining phases"
    break
  fi
done

say "--- clocks after ---"
rocm-smi --showclocks 2>&1 | grep -i sclk | head -10
say "campaign done, status=$status"
exit $status
