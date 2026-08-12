#!/usr/bin/env bash
# exp_34 GPU ladder step 0: take the lease, pin the checkout, inventory the
# harness runners that exp_35 just landed.
set -uo pipefail
echo "############ GPU occupancy (the check that works) ############"
rocm-smi --showpids 2>&1 | head -30

echo
echo "############ PIN ############"
cd $HOME/Distributed-HipKittens
git fetch --all -q
git reset -q --hard 291dfa08
git log --oneline -1
git status --porcelain | head
echo "--- the two owned files at the pin ---"
grep -n "K0P6_MPS_SRC_REV\|K0P6_MPS_ASCALE_TM" distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip | head
echo "--- mode 14 present? ---"
grep -c "kModeCoarseReady\|kCoarseKeepDrainBit" distributed-kernels/fused_moe/moe_mps_adapter.cuh

echo
echo "############ harness runners available ############"
ls -l $HOME/Distributed-HipKittens/distributed-kernels/fused_moe/overnight/aug11/tools/ | grep -E "e35_|e33_" | head -30
echo
echo "############ run_campaign.sh knobs ############"
sed -n '90,170p' $HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh
echo "===DONE==="
