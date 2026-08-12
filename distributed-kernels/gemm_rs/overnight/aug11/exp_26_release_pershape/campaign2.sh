#!/usr/bin/env bash
# Round 2. Round 1 established the mechanism and disqualified the spelling:
#
#   * rg2c (RELEASE_GROUP=2, incumbent rule -> rgroup 2 on shape 5) is the
#     fastest arm on 8192x4096x14336 in BOTH allocation orders, 639.3 and 641.7
#     against ps0's 661.3 and 664.3 -- pooled ranges disjoint, -3.3%. It is not
#     shippable, because rgroup 2 on 8192x8192x29568 costs +7.3%.
#   * ps1 (min(RG, tiles/CTA)) computes the SAME rgroup as rg2c on shape 5 and
#     did not get the same time: -2.4% pooled, inside that shape's 2.4% null
#     arm, and it moved two rows whose group size it does not change (+1.4% on
#     shape 1, +1.0% on shape 6). Its resource tuple explains why: +2 VGPRs on
#     five of seven instantiations.
#
# ps2 is the same rule spelled as a select over compile-time literals. If the
# codegen hypothesis is right it lands on rg2c's shape-5 time and ps0's shape-6
# time, with ps0's register budget. If it lands anywhere else the hypothesis is
# wrong and the incumbent stays.
#
# Round 1 used 7 rounds x 50 iters, which is ~4 s of actual timing per pass --
# far too few samples for deltas this size next to a 2.4% floor. This runs
# 25 x 100 in both allocation orders.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
ROUNDS=${1:-25}
ITERS=${2:-100}
mkdir -p "$D/logs"

# tools/gpu_lease.sh is currently on the node with CRLF line endings and does
# not run ("set: pipefail: invalid option name", "syntax error near $'{\r'").
# It belongs to another agent, so rather than edit a file this experiment does
# not own, run a CR-stripped copy: same script, same lock directory, same log,
# no shared file touched. If the shared copy is repaired later this keeps
# working unchanged.
LEASE=/tmp/gpu_lease_exp26.sh
tr -d '\r' < "$ON/tools/gpu_lease.sh" > "$LEASE"

stage() { printf '\n########## %s  (%s) ##########\n' "$1" "$(date -u +%FT%TZ)"; }

# IN THE CONTAINER. hipcc on the host has no pybind11 and picks up python3.10;
# round 2's first attempt ran build_arms.sh on the host, which deleted all five
# arm .so files (the builder removes its target first, deliberately, so that
# "the file exists" is a real build result) and then failed to rebuild any of
# them. Everything that compiles goes through dhk-gemmrs.
stage "rebuild all five arms (CPU only, no lease needed)"
docker exec dhk-gemmrs bash "$D/build_arms.sh" ps0 ps1 ps2 rg2c ps0b 2>&1 \
  | tee "$D/logs/build_arms2.log"
grep -q 'ARM MODULES BUILT' "$D/logs/build_arms2.log" || {
  echo "ABORT: arm build failed"; exit 1; }

stage "acquire the lease"
bash "$LEASE" acquire exp_26 10800 || { echo "ABORTED: no lease"; exit 1; }
trap 'bash "$LEASE" release exp_26 2>&1 | head -2' EXIT

for dir in 0 1; do
  name="ab2_$([ "$dir" = 0 ] && echo fwd || echo rev)"
  stage "A/B pass $name (reverse=$dir), $ROUNDS rounds x $ITERS iters"
  docker exec -w $ON/harness -e AB_TAG="$name" -e AB_CANDIDATE=ps2 dhk-gemmrs \
    timeout 7200 python3 -u "$D/ab_pershape.py" "$ROUNDS" "$ITERS" 800 "$dir" \
    2>&1 | tee "$D/logs/$name.log"
  echo "$name exit: ${PIPESTATUS[0]}"
done

stage "done"
echo "CAMPAIGN2 COMPLETE"
