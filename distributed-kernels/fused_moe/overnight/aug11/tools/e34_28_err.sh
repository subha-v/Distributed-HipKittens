#!/usr/bin/env bash
set -uo pipefail
DHK=$HOME/e34/DHK
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$HOME/e34/isa; mkdir -p "$OUT"
cd "$DHK"
echo "### git status of scratch clone"; git status --porcelain | head; git log --oneline -1
echo "### toolchain"; which hipcc; ls /opt/rocm*/lib/llvm/bin/llvm-objdump
FLAGS="--offload-arch=gfx950 -std=c++20 -O3 -DKITTENS_CDNA4
 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1
 -DK0P6GM_G=3 -DN2GM_G=3
 -I$DHK/include -I$DHK/distributed-kernels/fused_moe
 -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
 -I$MR -I$MR/include -I$MR/src
 -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include"
echo "### FULL compile output"
hipcc $FLAGS --genco distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
  -o "$OUT/B.hsaco" 2>&1 | grep -v "^Function Name\|^ *TotalSGPRs\|^ *VGPRs\|^ *AGPRs\|^ *Scratch\|^ *Dynamic\|^ *Occupancy\|^ *SGPRs Spill\|^ *VGPRs Spill\|^ *LDS" | head -60
echo "===DONE==="
