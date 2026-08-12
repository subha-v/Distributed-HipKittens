#!/usr/bin/env bash
# Everything exp_26 still needs from the GPU, under ONE properly acquired lease.
#
# This experiment's first two GPU runs (the gate ladder and the M9 gate) were
# launched WITHOUT tools/gpu_lease.sh, which is a discipline failure and it had
# a cost: the M9 run took a memory access fault, wedged in driver teardown, and
# held all 8 GPUs long enough that exp_21 and exp_24 both aborted their acquire
# at 10:06Z and 10:11Z. Disclosed in result.md. Everything below is leased.
#
#   campaign.sh [rounds] [iters]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LEASE=$ON/tools/gpu_lease.sh
ROUNDS=${1:-7}
ITERS=${2:-50}
mkdir -p "$D/logs"

stage() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date -u +%FT%TZ)"; }
release() { bash "$LEASE" release exp_26 2>&1 | head -2; }

stage "acquire the lease (waiting behind whoever holds it)"
bash "$LEASE" acquire exp_26 10800 || { echo "CAMPAIGN ABORTED: no lease"; exit 1; }
trap release EXIT

# ---------------------------------------------------------------------------
# 1. M9 re-aimed at shape 5 -- the ONLY shape whose group size this change
#    moves, and the one the stock case list no longer covers since exp_14
#    retiled row 2 to 1 tile per CTA. Ordered first because it is the gate that
#    is specific to this change.
# ---------------------------------------------------------------------------
stage "M9 (shape-5 aimed): poisoned heap, bitwise golden, CTRL_PUBLISH_EARLY"
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u "$D/m9_shape5.py" 2>&1 | tee "$D/logs/m9_shape5.log"
echo "m9_shape5 exit: ${PIPESTATUS[0]}"

# ---------------------------------------------------------------------------
# 2. The paired A/B in both allocation orders -- the decisive measurement.
# ---------------------------------------------------------------------------
for dir in 0 1; do
  name="ab_$([ "$dir" = 0 ] && echo fwd || echo rev)"
  stage "A/B pass $name (reverse=$dir), $ROUNDS rounds x $ITERS iters"
  docker exec -w $ON/harness -e AB_TAG="$name" dhk-gemmrs timeout 5400 \
    python3 -u "$D/ab_pershape.py" "$ROUNDS" "$ITERS" 600 "$dir" 2>&1 \
    | tee "$D/logs/$name.log"
  echo "$name exit: ${PIPESTATUS[0]}"
done

# ---------------------------------------------------------------------------
# 3. The stock M9 gate, completing the mandated ladder. It faulted at 10:05Z on
#    2048x2880x2880 -- a 1-tile-per-CTA row where this change is a behavioural
#    no-op -- so this re-run is also the test of whether that fault reproduces.
# ---------------------------------------------------------------------------
stage "M9 (stock case list), re-run after the 10:05Z fault"
docker exec -w $ON/harness dhk-gemmrs timeout 5400 \
  python3 -u m9_stale_slot.py 2>&1 | tee "$D/logs/m9_full.log"
echo "m9_full exit: ${PIPESTATUS[0]}"

stage "done"
echo "CAMPAIGN COMPLETE"
