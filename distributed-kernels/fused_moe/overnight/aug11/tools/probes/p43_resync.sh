#!/usr/bin/env bash
# exp_24: resync the node checkout onto origin (the containers bind-mount it
# read-only, so it IS the arm) and confirm the exp_24 commit landed.
set -uo pipefail
DHK=$HOME/Distributed-HipKittens
echo "=== before"
git -C "$DHK" rev-parse --short HEAD
git -C "$DHK" status --porcelain | head -5
echo "=== fetch + reset"
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard origin/codex/distributed-hipkittens-scaffold
echo "=== after"
git -C "$DHK" log --oneline -1
git -C "$DHK" rev-parse HEAD
grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | tail -1
echo "=== exp_24 symbols present in the node's adapter?"
grep -c 'skip_dead_part_zero\|throttle_depth_sel\|throttle_enabled\|scale_transpose_row' \
  "$DHK/distributed-kernels/fused_moe/moe_mps_adapter.cuh"
echo "=== node idle"
pgrep -af 'torchrun|mpirun' | head
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | sed -n '6,10p'
exit 0
