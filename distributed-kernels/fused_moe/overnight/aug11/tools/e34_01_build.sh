#!/usr/bin/env bash
# exp_34 step 1: CPU-only genco build + resource remark capture, from the private
# scratch clone at ~/e34/DHK. NO GPU WORK. Never touches ~/Distributed-HipKittens.
#
# Two arms from ONE tree, so the resource comparison is honest:
#   A (baseline) = the scratch clone at HEAD ca5b683f, pristine (mode 14 absent)
#   B (candidate) = the same tree with the exp_34 patch applied
# Both are built with identical flags; the gate is that B's resource tuple equals
# A's, i.e. mode 14 present-but-unused costs the existing modes nothing.
set -uo pipefail
H=$HOME
SC=$H/e34
mkdir -p "$SC/out" "$SC/base/distributed-kernels/fused_moe"

echo "=== arm A: pristine baseline copy of the two owned files ==="
cd "$SC/DHK"
git stash list >/dev/null 2>&1
git show ca5b683f:distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
  > "$SC/base/k0pf6gm_device_tile_mps.hip.orig"
git show ca5b683f:distributed-kernels/fused_moe/moe_mps_adapter.cuh \
  > "$SC/base/moe_mps_adapter.cuh.orig"
sha256sum "$SC/base/"*.orig

echo "=== candidate files as delivered ==="
sha256sum "$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
          "$SC/DHK/distributed-kernels/fused_moe/moe_mps_adapter.cuh"
echo "--- diff vs ca5b683f (diffstat) ---"
git diff --stat ca5b683f -- distributed-kernels/fused_moe/
git diff ca5b683f -- distributed-kernels/fused_moe/ > "$SC/out/e34.diff"
wc -l "$SC/out/e34.diff"

echo "=== build both arms in subha_k1 (CPU genco) ==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/e34
DHK=$SC/DHK
FM=$DHK/distributed-kernels/fused_moe
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
COMMON=(--genco --offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include
  -I$FM
  -I$K0/solution/hip/hkp
  -I$K0/prefill_opt/kernels
  -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)

# arm B = the tree as delivered
echo "### building B (exp_34 candidate)"
t0=$SECONDS
hipcc "${COMMON[@]}" "$FM/k0pf6gm_device_tile_mps.hip" -o $SC/out/B.hsaco \
  > $SC/out/B.log 2>&1
echo "B exit=$? secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/B.log)"

# arm A = the same tree with the two owned files reverted to ca5b683f
cp "$FM/k0pf6gm_device_tile_mps.hip" $SC/out/B_kernel.hip.keep
cp "$FM/moe_mps_adapter.cuh"        $SC/out/B_adapter.cuh.keep
cp $SC/base/k0pf6gm_device_tile_mps.hip.orig "$FM/k0pf6gm_device_tile_mps.hip"
cp $SC/base/moe_mps_adapter.cuh.orig        "$FM/moe_mps_adapter.cuh"
echo "### building A (pristine ca5b683f)"
t0=$SECONDS
hipcc "${COMMON[@]}" "$FM/k0pf6gm_device_tile_mps.hip" -o $SC/out/A.hsaco \
  > $SC/out/A.log 2>&1
echo "A exit=$? secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/A.log)"
cp $SC/out/B_kernel.hip.keep  "$FM/k0pf6gm_device_tile_mps.hip"
cp $SC/out/B_adapter.cuh.keep "$FM/moe_mps_adapter.cuh"
echo "### restored candidate files"
sha256sum "$FM/k0pf6gm_device_tile_mps.hip" "$FM/moe_mps_adapter.cuh"
'

echo "=== B errors (first 60 lines if any) ==="
grep -nE "error:" "$SC/out/B.log" | sed -n '1,60p' || echo "(none)"
echo "=== A errors ==="
grep -nE "error:" "$SC/out/A.log" | sed -n '1,20p' || echo "(none)"
