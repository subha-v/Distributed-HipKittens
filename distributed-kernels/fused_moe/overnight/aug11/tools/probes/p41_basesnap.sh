#!/usr/bin/env bash
# exp_24: snapshot the PRE-exp_24 kernel sources before the node checkout is
# reset onto the exp_24 commit, so the resource/ISA baseline stays buildable.
set -uo pipefail
DHK=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe
SNAP=$HOME/exp24-base/fused_moe
echo "=== node checkout state BEFORE snapshot"
git -C "$HOME/Distributed-HipKittens" rev-parse --short HEAD
grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$DHK/k0pf6gm_device_tile_mps.hip" | tail -1
rm -rf "$HOME/exp24-base"
mkdir -p "$SNAP"
cp "$DHK"/*.hip "$DHK"/*.cuh "$DHK"/*.cpp "$DHK"/*.hpp "$SNAP"/ 2>/dev/null
ls -1 "$SNAP"
echo "=== snapshot SRC_REV (must be 23)"
grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$SNAP/k0pf6gm_device_tile_mps.hip" | tail -1
echo
echo "=== node idle re-check"
pgrep -af 'torchrun|mpirun' | head
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | sed -n '4,12p'
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 | head -1
echo
echo "=== other GPU-job evidence: recent screen csvs / running batches"
ls -lt $HOME/overnight-scratch/*.csv 2>/dev/null | head -6
pgrep -af 'screen.sh|run_campaign' | head
exit 0
