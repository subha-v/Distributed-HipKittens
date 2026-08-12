#!/usr/bin/env bash
# exp_24 probe 1: locate hkp::zero_part_scale_transpose, confirm node state.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
DHK="$HOME/Distributed-HipKittens"

echo "=== node identity"
hostname
date -u +%FT%TZ
echo "=== DHK head"
git -C "$DHK" log --oneline -1
git -C "$DHK" rev-parse HEAD
git -C "$DHK" rev-parse --abbrev-ref HEAD
git -C "$DHK" status --porcelain | head -20
echo "=== gpu idle check"
pgrep -af 'torchrun|mpirun' | head
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | head -20
echo "=== lock dir"
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 | head -2

echo
echo "=== where does zero_part_scale_transpose live"
grep -rn "zero_part_scale_transpose" "$K0" --include=*.hpp --include=*.hip --include=*.cuh --include=*.cpp --include=*.h 2>/dev/null | head -40

echo
echo "=== hkp_quant.hpp candidates"
find "$HOME/amd-master" -name 'hkp_quant.hpp' 2>/dev/null | head -10
find "$HOME" -maxdepth 6 -name 'hkp_quant.hpp' 2>/dev/null | head -10
exit 0
