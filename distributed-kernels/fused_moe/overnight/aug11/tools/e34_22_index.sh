#!/usr/bin/env bash
# exp_34 condition 1, take 3: the EXACT index my publish/poll writes, and the
# kernel-side definitions of T_ext / T_loc_max / MAXTOK it depends on.
set -uo pipefail
D=$HOME/e34/DHK/distributed-kernels/fused_moe
KRN=$D/k0pf6gm_device_tile_mps.hip
ADP=$D/moe_mps_adapter.cuh

echo "### candidate tree:"; ls -l "$KRN" "$ADP"
echo
echo "################ publish + poll index sites (mode 14) ################"
grep -n -E "T_ext|self_slot|K0P6_MPS_ERR_M7DONE" "$KRN" | head -40
echo
echo "################ where T_ext is DEFINED in the kernel ################"
grep -n -E "T_ext *=" "$KRN" | head -20
echo
echo "################ T_loc_max + MAXTOK in the kernel ################"
grep -n -E "T_loc_max *=|MAXTOK" "$KRN" | head -20
echo
echo "################ the mode-14 entry guard, verbatim ################"
sed -n '755,790p' "$KRN"
echo
echo "################ the adapter validator, verbatim ################"
grep -n -B4 -A18 "mode > 14u" "$ADP"
echo "===DONE==="
