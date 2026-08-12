#!/usr/bin/env bash
# exp_26 step 1: snapshot a private, immutable build tree so B0/B1/B2 share
# everything except the phase-1 include. Never touches the shared checkout.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
DHKSRC=$HOME/Distributed-HipKittens

echo "===CAMPAIGN_CHECK==="
pgrep -af 'run_campaign|torchrun|mpirun' || echo "(no campaign / no gpu job)"

echo "===NODE_CHECKOUT_STATE==="
git -C "$DHKSRC" log --oneline -1
git -C "$DHKSRC" status --porcelain=v1 | head -30
echo "(dirty files above, if any)"

echo "===SNAPSHOT==="
rm -rf "$SC/dhk"
mkdir -p "$SC/dhk" "$SC/tu" "$SC/out" "$SC/incoming"
git -C "$DHKSRC" archive HEAD | tar -x -C "$SC/dhk"
echo "snapshot_head=$(git -C "$DHKSRC" rev-parse HEAD)"
sha256sum "$SC/dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
          "$SC/dhk/distributed-kernels/fused_moe/moe_mps_adapter.cuh" \
          "$SC/dhk/distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp"
echo "===P1_INCLUDE_LINE==="
grep -n 'n2_phase1_gm.cpp' "$SC/dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
grep -n 'K0P6_MPS_SRC_REV' "$SC/dhk/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
echo "===READY==="
ls -la "$SC"
