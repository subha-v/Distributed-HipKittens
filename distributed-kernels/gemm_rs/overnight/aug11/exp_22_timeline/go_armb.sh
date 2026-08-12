#!/usr/bin/env bash
# exp_22 arm (b): the phase ring, the tick calibration, and the integral check.
#
# The kernel changed under this experiment: exp_26 landed
# HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=2, which moves shape 5's rgroup from
# 1 to 2 -- and shape 5 is the shape being traced. So everything is rebuilt from
# the current source and the flag-OFF parity is re-asserted against the current
# incumbent tuple before a single microsecond is collected.
#
# Structure: compiling needs no GPU, so the parity gate and both builds happen
# OUTSIDE the lease. The lease covers only the launches. That keeps the hold
# short, which matters while exp_24 is re-measuring the ratchet.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
GEMM=$ON/..
E22=$ON/aug11/exp_22_timeline
LEASE_HELD=0

release() {
  if [ "$LEASE_HELD" = "1" ]; then
    bash "$ON/tools/gpu_lease.sh" release exp_22 || true
    LEASE_HELD=0
  fi
}
trap release EXIT INT TERM

echo "############ provenance: what exactly is being measured ############"
sha256sum "$GEMM/gemm_rs_mi300x.cpp" "$GEMM/gemm_rs_mi300x_constants.cuh" \
          "$ON/harness/build/gemm_rs_mi300x.so" 2>&1
echo "--- flag defaults in the committed source (TRACE must read 0) ---"
grep -n -A1 "ifndef HK_GEMM_RS_MI300X_TRACE\|ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE" \
  "$GEMM/gemm_rs_mi300x.cpp" | grep define

echo
echo "############ GATE 1 (CPU): flag-OFF resource-tuple parity, re-asserted ############"
# Re-assert, not re-derive: parity_check.py compares the two OFF arms to each
# other AND both to the incumbent tuple it has hard-coded. Skipped only if the
# verdict on disk is newer than the source it was taken from.
if [ "$E22/parity.json" -nt "$GEMM/gemm_rs_mi300x.cpp" ]; then
  echo "parity.json is newer than the source; re-asserting from disk"
else
  docker exec -w "$E22" dhk-gemmrs bash "$E22/parity_gate.sh" 2>&1 | tail -30
fi
docker exec -w "$E22" dhk-gemmrs python3 -c "
import json
d = json.load(open('$E22/parity.json'))
print('parity verdict:', d.get('verdict'), '| failures:', len(d.get('failures', [])))
for f in d.get('failures', [])[:8]: print('  ', f)
"

echo
echo "############ build the diagnostic module (flag ON) ############"
docker exec -w "$E22" dhk-gemmrs bash "$E22/build_trace.sh" || {
  echo "ABORT: diagnostic build failed"; exit 1; }
sha256sum "$E22/build/gemm_rs_mi300x_trace.so"

# An earlier capture of unknown build identity is sitting here. Archive rather
# than overwrite, so the new run's outputs cannot be confused with it and the
# old one stays available if the two ever need comparing.
PREV=$E22/prev_run_$(date +%H%M)
if ls "$E22"/events_ours_*.json >/dev/null 2>&1; then
  mkdir -p "$PREV"
  mv "$E22"/events_ours_*.json "$E22/tick_rate.json" "$E22/validation.json" \
     "$E22/timeline_bins.csv" "$PREV/" 2>/dev/null
  echo "archived the previous capture to $PREV"
fi

echo
echo "############ acquiring the lease (blocks; exp_24 may hold it) ############"
bash "$ON/tools/gpu_lease.sh" acquire exp_22 5400 || { echo "ABORT: no lease"; exit 1; }
LEASE_HELD=1
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -6
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -2
rocm-smi --showperflevel 2>/dev/null | grep -i perf | head -9 | tee "$E22/clocks_before.txt"

echo
echo "############ smoke: smallest shape, one launch, before the real run ############"
# trace_run.py has never executed. A two-minute smoke on s2 exercises every
# risky path (module swap, ring alloc, the appended argument, ring decode)
# without discovering a plumbing bug forty minutes into the real capture.
setsid timeout 900 docker exec -w "$E22" dhk-gemmrs \
  python3 -u "$E22/trace_run.py" --shapes s2 --launches 1 --warmup-ms 300 \
  --outdir "$E22/smoke"
smoke_rc=$?
echo "smoke exit=$smoke_rc"
if [ "$smoke_rc" != "0" ]; then
  echo "ABORT: smoke failed; releasing the lease rather than holding it while debugging"
  exit 1
fi

echo
echo "############ arm (b): s5 (the figure) + s6 + s2 for the tick regression ############"
setsid timeout 3000 docker exec -w "$E22" dhk-gemmrs \
  python3 -u "$E22/trace_run.py" --shapes s5,s6,s2 --launches 3
echo "arm (b) exit=$?"

rocm-smi --showperflevel 2>/dev/null | grep -i perf | head -9 | tee "$E22/clocks_after.txt"
release

echo
echo "############ bin + validate integrals (CPU, no lease) ############"
docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/bin_timeline.py" \
  "$E22/events_ours_s5.json" "$E22/events_b0_reference.json" \
  --csv "$E22/timeline_bins.csv" --json "$E22/validation.json"
echo "binning exit=$?"
echo "===== DONE ====="
