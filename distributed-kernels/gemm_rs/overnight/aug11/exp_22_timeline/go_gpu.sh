#!/usr/bin/env bash
# exp_22 GPU phase, under the node lease. Arms (a) and (b) only; arm (c) is a
# separate dispatch and must not start in this lease.
#
# The lease is not a formality: a dirty-node check alone lets two agents observe
# a clean node in the same second and both launch, which corrupts both sets of
# numbers invisibly. tools/gpu_lease.sh serializes with an atomic mkdir, does
# the wait-for-drain itself, and (since the exp_26 VM_L2_PROTECTION_FAULT)
# distinguishes live tenants from stale KFD entries. It is expected to report
# one stale pid; that is documented, not a reason to stop, and the stale task
# must NEVER be signalled -- it sits in exit_mm with no address space, so
# SIGTERM and SIGKILL are both no-ops and attempting them is how a stale entry
# becomes a wedged node.
#
# The release is TRAPPED so a failure anywhere below cannot strand the lease.
#
#   go_gpu.sh [sanity|a|b|all]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
WHAT=${1:-all}
LEASE_HELD=0

release() {
  if [ "$LEASE_HELD" = "1" ]; then
    bash "$ON/tools/gpu_lease.sh" release exp_22 || true
    LEASE_HELD=0
  fi
}
trap release EXIT INT TERM

echo "===== acquiring the GPU lease (blocks; drains the node itself) ====="
bash "$ON/tools/gpu_lease.sh" acquire exp_22 5400 || { echo "ABORT: no lease"; exit 1; }
LEASE_HELD=1
bash "$ON/tools/gpu_lease.sh" status 2>/dev/null | head -12

echo
echo "===== clocks pinned, recorded around every measurement ====="
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -3
# --showclocks reports the INSTANTANEOUS clock, which reads ~120 MHz whenever
# the query lands on an idle GPU even with the perf level forced. --showperflevel
# is what says whether the pin took, so record both.
rocm-smi --showperflevel --showclocks 2>/dev/null \
  | tee "$E22/clocks_before.txt" | grep -Ei "perf|sclk" | head -20

# ---------------------------------------------------------------------------
# NODE SANITY, FIRST AND GATING. exp_22 is the first campaign after the
# incident. Re-measure a value we already know before trusting anything else:
# the full M7 vector at the flag-OFF production build, against the current
# best-of-arm 62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63 us. Shape 5 is
# the gate because it is this figure's shape.
#
# The production .so in harness/build is used AS IS and is not rebuilt --
# harness/ is read-only to this experiment, and the binary that produced the
# reference vector is the binary that should reproduce it.
# ---------------------------------------------------------------------------
echo
echo "############ node sanity: M7 vector at the ratchet build ############"
# EXP22_REUSE_M7=1 re-scores an M7 run already taken under a lease in this
# session rather than paying for it twice. The bench is the expensive part; the
# comparison is free, and the first pass's comparison was against the wrong
# statistic (best-of-arm rather than the mean-geomean M7 actually prints).
if [ "${EXP22_REUSE_M7:-0}" = "1" ] && [ -f "$E22/m7_results.json" ]; then
  echo "reusing $E22/m7_results.json (taken under this session's lease):"
  ls -l "$E22/m7_results.json"
else
  setsid timeout 2400 docker exec -w "$E22" dhk-gemmrs \
    env PYTHONPATH="$ON/harness" \
    python3 -u "$ON/harness/m7_bench.py" 3 50 2>&1 | tail -32
fi
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/sanity_check.py" \
  "$E22/m7_results.json" --json "$E22/sanity.json"
sanity_rc=$?
if [ "$sanity_rc" != "0" ]; then
  echo "ABORT: node sanity FAILED. Not collecting timeline data on a node whose"
  echo "       known values do not reproduce -- that matters more than the figure."
  exit 1
fi
[ "$WHAT" = "sanity" ] && { echo "sanity only, stopping"; exit 0; }

if [ "$WHAT" = "a" ] || [ "$WHAT" = "all" ]; then
  echo
  echo "############ arm (a): reference GEMM+RCCL kernel trace, shape 5 ############"
  setsid timeout 1800 bash "$E22/b0_capture.sh" s5 8192 4096 14336 1 7168
  echo "arm (a) exit=$?"
fi

if [ "$WHAT" = "b" ] || [ "$WHAT" = "all" ]; then
  echo
  echo "############ arm (b): build the diagnostic module ############"
  docker exec -w "$E22" dhk-gemmrs bash "$E22/build_trace.sh" || {
    echo "ABORT: diagnostic build failed"; exit 1; }

  echo
  echo "############ arm (b): phase ring + in-situ tick calibration ############"
  # s5 is the figure; s6 and s2 exist only to give the tick regression two more
  # very different durations, and cost three launches each.
  setsid timeout 2400 docker exec -w "$E22" dhk-gemmrs \
    python3 -u "$E22/trace_run.py" --shapes s5,s6,s2 --launches 3
  echo "arm (b) exit=$?"
fi

rocm-smi --showperflevel --showclocks 2>/dev/null \
  | tee "$E22/clocks_after.txt" | grep -Ei "perf|sclk" | head -20

echo
echo "############ bin + validate (CPU) ############"
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/bin_timeline.py" \
  "$E22"/events_ours_s5.json "$E22"/events_b0_reference.json \
  --csv "$E22/timeline_bins.csv" --json "$E22/validation.json"
echo "binning exit=$?"

echo "===== DONE (lease released by the exit trap) ====="
