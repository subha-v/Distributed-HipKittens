#!/usr/bin/env bash
# exp_34 condition 3 (in container subha_k1, CPU-only): prove a system-scope
# invalidate precedes the first payload load in the mode-14 m8_batch.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
DHK=$H/e34/DHK
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/e34/isa; mkdir -p $OUT
OBJ=/opt/rocm/lib/llvm/bin/llvm-objdump
BND=/opt/rocm/llvm/bin/clang-offload-bundler
[ -x $BND ] || BND=/opt/rocm/lib/llvm/bin/clang-offload-bundler
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -I$DHK/include -I$DHK/distributed-kernels/fused_moe
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
SRC=$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

echo "### (0) genco build"
hipcc "${COMMON[@]}" --genco $SRC -o $OUT/B.hsaco > $OUT/B.log 2>&1
echo "exit=$? errors=$(grep -cE \"error:\" $OUT/B.log)"; ls -l $OUT/B.hsaco

echo "### (1) bundle inspection"
file $OUT/B.hsaco
$OBJ --offloading $OUT/B.hsaco 2>&1 | head -8

echo "### (2) unbundle -> objdump"
ok=0
for T in hipv4-amdgcn-amd-amdhsa--gfx950 hip-amdgcn-amd-amdhsa--gfx950; do
  if $BND --unbundle --type=o --targets=$T -input=$OUT/B.hsaco -output=$OUT/B.co 2>/dev/null; then
    echo "unbundled: $T"; ok=1; break; fi
done
[ $ok -eq 1 ] || { echo "unbundle FAILED, objdumping bundle directly"; cp $OUT/B.hsaco $OUT/B.co; }
$OBJ -d --mcpu=gfx950 $OUT/B.co > $OUT/B.dis 2>$OUT/B.dis.err
echo "dis lines=$(wc -l < $OUT/B.dis) err=$(head -1 $OUT/B.dis.err)"
echo "sanity: v_mfma count=$(grep -c v_mfma $OUT/B.dis)  buffer_inv count=$(grep -c buffer_inv $OUT/B.dis)"

echo "### (3) every buffer_inv window (3 before / 12 after)"
for L in $(grep -n buffer_inv $OUT/B.dis | cut -d: -f1); do
  echo "===== dis:$L ====="
  sed -n "$((L-3)),$((L+12))p" $OUT/B.dis
done

echo "### (4) source attribution via -S -gline-tables-only"
hipcc "${COMMON[@]}" -gline-tables-only --cuda-device-only -S $SRC -o $OUT/B.s > $OUT/Bs.log 2>&1
echo "exit=$? asm lines=$(wc -l < $OUT/B.s)"
grep -nE "^[[:space:]]*\.file[[:space:]]+[0-9]+" $OUT/B.s | head -8
awk "
  /\.loc[ \t]+[0-9]+[ \t]+[0-9]+/ { split(\$0,a,\" \"); fid=a[2]; ln=a[3] }
  /buffer_inv/ { printf \"buffer_inv  file=%s srcline=%s  |%s\n\", fid, ln, \$0 }
" $OUT/B.s
' 2>&1
echo "===DONE==="
