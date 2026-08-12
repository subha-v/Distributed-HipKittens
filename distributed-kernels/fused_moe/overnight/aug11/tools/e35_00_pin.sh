#!/usr/bin/env bash
# exp_35 step 0: pin the node checkout to the exp_35 arm SHA and verify the arm.
set -uo pipefail
DHK=$HOME/Distributed-HipKittens
PIN=ca5b683f

echo "===PIN==="
git -C "$DHK" fetch --all -q 2>&1 | tail -3
git -C "$DHK" reset -q --hard "$PIN" 2>&1 | tail -3
git -C "$DHK" log -1 --format='HEAD=%H%n SUBJ=%s%n DATE=%ci'
echo "--- dirty files (must be empty) ---"
git -C "$DHK" status --porcelain | head -20
echo "(end dirty)"

echo "===LOCATE_SOURCES==="
find "$DHK/distributed-kernels" -name 'moe_mps_adapter.cuh' -o -name 'k0pf6gm_device_tile_mps.hip' | sort

echo "===ASCALE_TM==="
grep -rn 'K0P6_MPS_ASCALE_TM' "$DHK/distributed-kernels" | head -20

echo "===MPS_SRC_REV==="
grep -rn 'K0P6_MPS_SRC_REV' "$DHK/distributed-kernels" | head -10

echo "===SCRATCH==="
mkdir -p "$HOME/e35/raw"
ls -d "$HOME/e35"
echo "===HARNESS==="
ls "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/" | head -30
ls "$HOME/tools/" 2>&1 | head -20
