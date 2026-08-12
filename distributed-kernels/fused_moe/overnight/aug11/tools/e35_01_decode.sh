#!/usr/bin/env bash
# exp_35 step 1: derive the exact bit layout of the K0_MPS_CFG `g` field, and in
# particular the throttle-depth selector under mask 0x300 and whether an
# "unthrottled/disabled" encoding exists.
set -uo pipefail
DHK=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe
ADP=$DHK/moe_mps_adapter.cuh
MPS=$DHK/k0pf6gm_device_tile_mps.hip

echo "===HITS_ADAPTER==="
grep -nE 'throttle|0x300|0x1ff|0x1FF|0xff|0xFF|group_n|cfg|K0_MPS_CFG|MPS_TRACE' "$ADP" | head -120

echo
echo "===HITS_MPS_HIP==="
grep -nE 'throttle|0x300|>> *34|34\)|group_n|skip_dead|MPS_TRACE|decode' "$MPS" | head -140

echo
echo "===HOST_BRIDGE_PARSE==="
HB=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/mps_host_bridge.cpp
grep -nE 'K0_MPS_CFG|throttle|group|flush_rows|sscanf|MPS_TRACE|mode' "$HB" | head -80

echo
echo "===GREP_TRACE_PRINTS==="
grep -rn 'MPS TRACE\|K0_MPS_TRACE\|MPS CFG' "$DHK" "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe" 2>/dev/null | grep -v overnight/ | head -40
# end
