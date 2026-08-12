#!/usr/bin/env bash
# exp_34: the MPS cfg word bit layout, to find a FREE bit for the
# drain-retention selector (must be in the same word the hooks already load).
set -uo pipefail
D=$HOME/e34/DHK/distributed-kernels/fused_moe
ADP=$D/moe_mps_adapter.cuh
KRN=$D/k0pf6gm_device_tile_mps.hip

echo "################ cfg pack / unpack ################"
grep -n -B6 -A28 "pack_config|encode_config|inline .*pack" "$ADP" | head -80
echo
echo "################ every named bit constant ################"
grep -n -E "kRemoteAccum[A-Za-z]*Bit|Bit *=|<< *[0-9]+;" "$ADP" | head -40
echo
echo "################ group_slices semantics comments ################"
grep -n -E "group_slices" "$ADP" | head -40
echo
echo "################ how the harness passes K0_MPS_CFG -> desc word ################"
grep -rn -E "MPS_CFG|K0_MPS_CFG" $HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py | head -25
echo "===DONE==="
