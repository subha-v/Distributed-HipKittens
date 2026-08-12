#!/usr/bin/env bash
# exp_34 ladder step 1: build AT THE PIN (read-only use of the pinned checkout),
# capture the resource remark, and census the ISA including the MFMA spans.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
T=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/e34/gate; mkdir -p $OUT
OBJ=/opt/rocm/lib/llvm/bin/llvm-objdump
BND=/opt/rocm/llvm/bin/clang-offload-bundler
[ -x $BND ] || BND=/opt/rocm/lib/llvm/bin/clang-offload-bundler
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$T/include -I$T/distributed-kernels/fused_moe
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
echo "### source identity"
grep -n "K0P6_MPS_SRC_REV \|define K0P6_MPS_ASCALE_TM" $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip | head -4
sha256sum $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip $T/distributed-kernels/fused_moe/moe_mps_adapter.cuh

echo "### build"
hipcc "${COMMON[@]}" --genco $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
   -o $OUT/PIN.hsaco > $OUT/PIN.log 2>&1
echo "exit=$? errors=$(grep -cE \"error:\" $OUT/PIN.log)"
grep -A11 "Function Name: k0pf6gm_mps_mega" $OUT/PIN.log | \
  grep -E "TotalSGPRs|VGPRs:|AGPRs|ScratchSize|Occupancy|Spill|LDS Size" | sed "s/.*remark: *//;s/ \[-Rpass.*//"

echo "### ISA census"
$BND --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
   -input=$OUT/PIN.hsaco -output=$OUT/PIN.co 2>/dev/null
$OBJ -d --mcpu=gfx950 $OUT/PIN.co > $OUT/PIN.dis 2>/dev/null
echo "dis=$(wc -l < $OUT/PIN.dis) mfma=$(grep -c v_mfma $OUT/PIN.dis) pk_add=$(grep -c flat_atomic_pk_add_bf16 $OUT/PIN.dis) inv_sc0sc1=$(grep -c \"buffer_inv sc0 sc1\" $OUT/PIN.dis) scratch_ops=$(grep -cE \"scratch_(load|store)\" $OUT/PIN.dis)"

echo "### scratch ops INSIDE either MFMA span (must be 0)"
python3 - <<PY
import re
lines=[l for l in open("$OUT/PIN.dis")]
idx=[i for i,l in enumerate(lines) if "v_mfma" in l]
# contiguous spans of mfma activity: group mfma instrs whose gaps are < 400 lines
spans=[]; start=idx[0]; prev=idx[0]
for i in idx[1:]:
    if i-prev>400: spans.append((start,prev)); start=i
    prev=i
spans.append((start,prev))
print("mfma spans:", [(a,b,sum(1 for j in idx if a<=j<=b)) for a,b in spans])
for a,b in spans:
    n=sum(1 for l in lines[a:b+1] if re.search(r"scratch_(load|store)",l))
    print(f"  span {a}-{b}: scratch ops inside = {n}")
PY
' 2>&1
echo "===DONE==="
