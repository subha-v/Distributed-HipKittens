#!/usr/bin/env bash
# exp_24: build the UNMODIFIED node checkout with the identical command, then
# disassemble both and diff the resource/ISA evidence.
set -uo pipefail

docker exec subha_k1 bash -lc '
set -uo pipefail
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
HOME=/home/subvadla
STAGE=$HOME/exp24-stage/fused_moe
BASE=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe
OUTD=$HOME/exp24-stage/build
mkdir -p "$OUTD"; cd "$OUTD"

build () {   # $1 = -I override dir (may be empty), $2 = source, $3 = out tag
  local iov="$1" src="$2" tag="$3" extra=""
  [ -n "$iov" ] && extra="-I$iov"
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    $extra \
    -I$HOME/Distributed-HipKittens/include \
    -I$BASE \
    -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp \
    -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels \
    -I$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    "$src" -o "$OUTD/${tag}.hsaco" 2> "$OUTD/${tag}.log"
  echo "  $tag rc=$?"
}

echo "=== BASELINE build (node checkout, SRC_REV $(grep -oE "K0P6_MPS_SRC_REV [0-9]+" $BASE/k0pf6gm_device_tile_mps.hip | tail -1 | awk "{print \$2}"))"
build "" "$BASE/k0pf6gm_device_tile_mps.hip" base
echo "=== EXP24 build (staged, SRC_REV $(grep -oE "K0P6_MPS_SRC_REV [0-9]+" $STAGE/k0pf6gm_device_tile_mps.hip | tail -1 | awk "{print \$2}"))"
build "$STAGE" "$STAGE/k0pf6gm_device_tile_mps.hip" exp24

echo
echo "=== resource tuple: base | exp24"
for t in base exp24; do
  echo "--- $t"
  grep -oE "(TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]|Occupancy \[waves/SIMD\]): [0-9]+" "$OUTD/${t}.log" | sort -u
done

echo
echo "=== hsaco sizes"
ls -l "$OUTD"/base.hsaco "$OUTD"/exp24.hsaco

echo
echo "=== disassemble"
for t in base exp24; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 "$OUTD/${t}.hsaco" > "$OUTD/${t}.s" 2>/dev/null \
    || /opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx950 "$OUTD/${t}.hsaco" > "$OUTD/${t}.s"
  echo "$t: $(wc -l < "$OUTD/${t}.s") lines"
done

echo
echo "=== s_waitcnt vmcnt(N) literal census (N != 0)"
for t in base exp24; do
  echo "--- $t"
  grep -oE "s_waitcnt +vmcnt\([0-9]+\)" "$OUTD/${t}.s" | sort | uniq -c | sort -k2,2V
done

echo
echo "=== MFMA census"
for t in base exp24; do
  printf "%s: total v_mfma = %s\n" "$t" "$(grep -cE "^\s+v_mfma" "$OUTD/${t}.s")"
  grep -oE "v_mfma_[a-z0-9_]+" "$OUTD/${t}.s" | sort | uniq -c
done

echo
echo "=== scratch op census"
for t in base exp24; do
  printf "%s: scratch_load=%s scratch_store=%s\n" "$t" \
    "$(grep -cE "scratch_load" "$OUTD/${t}.s")" \
    "$(grep -cE "scratch_store" "$OUTD/${t}.s")"
done

echo
echo "=== global_atomic_pk_add_bf16 / cmpswap census"
for t in base exp24; do
  printf "%s: pk_add_bf16=%s atomic_cmpswap=%s\n" "$t" \
    "$(grep -cE "global_atomic_pk_add_bf16" "$OUTD/${t}.s")" \
    "$(grep -cE "atomic_cmpswap" "$OUTD/${t}.s")"
done
'
exit 0
