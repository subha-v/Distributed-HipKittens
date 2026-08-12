#!/usr/bin/env bash
# Round 3: the same four arms through an instrument whose per-round arm order is
# SHUFFLED rather than rotated, and with enough rounds to make the paired sign
# test mean something.
#
# Round 2's paired analysis showed the null arm is not centred: ps0b beat ps0 by
# a median 2.29% on shape 6 in 57 of 64 paired rounds, on identical behaviour.
# A rotation pins the relative spacing of any two arms, so an order effect is a
# constant offset on that pair rather than noise that averages out. If the
# shuffle is the right diagnosis, ps0b's paired median collapses toward zero on
# every shape -- and only then does any other arm's paired median mean anything.
#
# rg2c is kept purely as the POSITIVE CONTROL. It is the one arm with a
# different rgroup on 8192x8192x29568 (2 against 4), it came out +3.68% there
# against a -2.29% null in round 2, and it is 4-for-4 in sign across the passes.
# If the shuffled instrument cannot still see rg2c on shape 6, it has no power
# over release granularity at all and nothing it says about shape 5 counts.
#
# ps1 is dropped: it is disqualified on its resource tuple (+2 VGPRs on five of
# seven instantiations), whatever its timing.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
ROUNDS=${1:-40}
ITERS=${2:-100}
mkdir -p "$D/logs"

LEASE=/tmp/gpu_lease_exp26.sh
tr -d '\r' < "$ON/tools/gpu_lease.sh" > "$LEASE"

stage() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date -u +%FT%TZ)"; }

stage "acquire the lease"
bash "$LEASE" acquire exp_26 10800 || { echo "ABORTED: no lease"; exit 1; }
trap 'bash /tmp/gpu_lease_exp26.sh release exp_26 2>&1 | head -2' EXIT

for dir in 0 1; do
  name="ab3_$([ "$dir" = 0 ] && echo fwd || echo rev)"
  stage "A/B pass $name (alloc order reverse=$dir), $ROUNDS x $ITERS, shuffled"
  docker exec -w $ON/harness -e AB_TAG="$name" -e AB_CANDIDATE=ps2 \
    -e AB_ARMS="ps0,ps2,rg2c,ps0b" dhk-gemmrs \
    timeout 7200 python3 -u "$D/ab_pershape.py" "$ROUNDS" "$ITERS" 800 "$dir" \
    2>&1 | tee "$D/logs/$name.log"
  echo "$name exit: ${PIPESTATUS[0]}"
done

stage "paired analysis over round 3 only"
python3 "$D/paired.py" "$D/logs" --only ab3 2>&1 | tee "$D/logs/paired_r3.txt"

stage "done"
echo "CAMPAIGN3 COMPLETE"
