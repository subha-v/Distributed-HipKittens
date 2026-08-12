#!/usr/bin/env bash
# exp_36 step 0: pin the node checkout, verify the ascale-TM guard, and find
# out whether the token count T is reachable through the campaign driver.
set -u
DHK="$HOME/Distributed-HipKittens"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
RC="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"
MPS="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"

git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard b5215081
echo "HEAD    : $(git -C "$DHK" rev-parse HEAD)"
git -C "$DHK" log --oneline -1
echo "dirty   : [$(git -C "$DHK" status --porcelain | head -3 | tr '\n' ';')]"
echo "ASCALE  : $(grep -n 'define K0P6_MPS_ASCALE_TM' "$MPS" | head -2)"
echo "SRC_REV : $(grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$MPS" | tail -1)"
echo "POISON  : $(grep -n 'K0_MOK_POISON_OUT' "$MPS" | head -2)"

echo "=== run_campaign.sh: token/shape env ==="
grep -n 'K0_T\b\|K0_T=\|MAXTOK\|K0_MAXTOK\|K0_BLK\|K0_WARP\|T_LOC_MAX\|PADMAX' "$RC"
echo "=== run_campaign.sh: forwarded env list ==="
grep -n -- '-e ' "$RC" | head -80
mkdir -p "$HOME/e36"
exit 0
