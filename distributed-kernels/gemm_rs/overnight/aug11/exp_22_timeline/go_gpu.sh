#!/usr/bin/env bash
# exp_22 GPU phase, under the node lease. Run only when the orchestrator hands
# the GPU over.
#
# The lease is not a formality: a dirty-node check alone lets two agents observe
# a clean node in the same second and both launch, which corrupts both sets of
# numbers invisibly. tools/gpu_lease.sh serializes with an atomic mkdir and does
# the wait-for-drain itself. The release is TRAPPED so a failure anywhere below
# cannot strand the lease.
#
#   go_gpu.sh [a|b|all]
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

echo "===== clocks pinned, recorded around every measurement ====="
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -3
rocm-smi --showclocks 2>/dev/null | tee "$E22/clocks_before.txt" | head -12

if [ "$WHAT" = "a" ] || [ "$WHAT" = "all" ]; then
  echo
  echo "############ arm (a): reference GEMM+RCCL kernel trace, shape 5 ############"
  # b0_capture.sh has its own preflight; it runs INSIDE the lease we already
  # hold, so its dirty-node check should see a clean node.
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
  # very different durations, and cost one launch each.
  setsid timeout 2400 docker exec -w "$E22" dhk-gemmrs \
    python3 -u "$E22/trace_run.py" --shapes s5,s6,s2 --launches 3
  echo "arm (b) exit=$?"
fi

rocm-smi --showclocks 2>/dev/null | tee "$E22/clocks_after.txt" | head -12

echo
echo "############ bin + validate (CPU) ############"
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/bin_timeline.py" \
  "$E22"/events_*.json --csv "$E22/timeline_bins.csv" \
  --json "$E22/validation.json"
echo "binning exit=$?"

echo "===== DONE (lease released by the exit trap) ====="
