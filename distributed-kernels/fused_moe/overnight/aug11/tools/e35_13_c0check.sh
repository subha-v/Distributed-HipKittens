#!/usr/bin/env bash
# exp_35 step 5d: (a) read the negative control's verdict, (b) decide whether
# C=0 is a LEGAL mode-12 config, so a supplementary rung can hold the schedule
# fixed and vary ONLY the dedicated-comm-CTA count -- the direct test of the
# paper's "not by dedicating CTAs to communication" claim.
set -uo pipefail
ADP=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/moe_mps_adapter.cuh

echo "===NEG_CONTROL_LOG_TAIL==="
tail -20 "$HOME/e35/neg.log" 2>&1

echo
echo "===NEG_CONTROL_CSV==="
cat "$HOME/overnight-scratch/screen_e35neg.csv" 2>&1

echo
echo "===MODE_PREDICATES_THAT_FORBID_C0==="
sed -n '160,180p' "$ADP"
sed -n '305,336p' "$ADP"

echo
echo "===C0_LEGALITY_FOR_MODE_12==="
grep -nE 'reserved_comm_ctas == 0u|reserved_comm_ctas *== *0' "$ADP"
echo "--- service pool entry predicate in the kernel (does C=0 mean 'no pool')? ---"
grep -nE 'reserved_comm_ctas|service_dies' "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | head -20
