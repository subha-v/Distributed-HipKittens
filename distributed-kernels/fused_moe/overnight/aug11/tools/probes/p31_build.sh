#!/usr/bin/env bash
# exp_24: genco build of the STAGED mps kernel + resource usage + ISA evidence.
set -uo pipefail
STAGE=/home/subvadla/exp24-stage/fused_moe
OUTD=/home/subvadla/exp24-stage/build
mkdir -p "$OUTD"

docker exec subha_k1 bash -lc '
set -uo pipefail
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
STAGE=/home/subvadla/exp24-stage/fused_moe
OUTD=/home/subvadla/exp24-stage/build
HOME=/home/subvadla
cd "$OUTD"
echo "=== hipcc version"; hipcc --version | head -3
echo "=== genco build (staged)"
time hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -Rpass-analysis=kernel-resource-usage \
  -I$STAGE \
  -I$HOME/Distributed-HipKittens/include \
  -I$HOME/Distributed-HipKittens/distributed-kernels/fused_moe \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels \
  -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip \
  -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  $STAGE/k0pf6gm_device_tile_mps.hip \
  -o $OUTD/exp24_mps.gfx950.hsaco 2> $OUTD/exp24_build.log
rc=$?
echo "build rc=$rc"
echo "=== resource-usage lines"
grep -A12 "k0pf6gm_mps_mega" $OUTD/exp24_build.log | head -40
echo "=== any errors/warnings"
grep -nE "error|warning" $OUTD/exp24_build.log | head -40
echo "=== tail of build log"
tail -30 $OUTD/exp24_build.log
'
exit 0
