#!/usr/bin/env bash
# exp_36 step 4 (read-only): confirm the g-bit throttle decode and the legal
# C range for mode 12, so the sweep points are legal by construction.
set -u
DHK="$HOME/Distributed-HipKittens"
F="$DHK/distributed-kernels/fused_moe"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"

echo "=== throttle bit decode in the kernel/adapter ==="
grep -rn '0x300\|0x20\b\|throttle\|THROTTLE\|depth' "$F/k0pf6gm_device_tile_mps.hip" "$F/moe_mps_adapter.cuh" 2>/dev/null | head -40
echo
echo "=== host-side config validator (C bounds, mode list) ==="
grep -n '_parse_mps_config' -A 60 "$AB" | head -90
echo
echo "=== k0_mps_host_abi encode_config ==="
ABI="$(find "$K0" "$DHK" -name 'k0_mps_host_abi*' 2>/dev/null | head -3)"
echo "$ABI"
for f in $ABI; do echo "--- $f"; grep -n 'def encode_config' -A 60 "$f" | head -80; done
exit 0
