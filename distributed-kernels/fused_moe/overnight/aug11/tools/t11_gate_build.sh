#!/usr/bin/env bash
# t11: exp_26 STEP-2 GATE. Build the CURRENT kernel source two ways --
#   D  = donor include  (#include "n2_phase1_gm.cpp")
#   M0 = vendored include at mask 0
# and require the .text sections to be BYTE-IDENTICAL. exp_26 proved this at a
# different snapshot (96049dfa..., 166,656 B); the source has moved since
# (exp_21/24), so the gate is re-derived here rather than compared to that hash.
# Also builds masks 1/4/5 for the step-3 ladder's resource + scratch facts.
# CPU-ONLY (hipcc --genco in subha_k1). No GPU work.
set -uo pipefail
SC=$HOME/overnight-scratch/e26act
DHK=$HOME/Distributed-HipKittens
FM=$DHK/distributed-kernels/fused_moe
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
mkdir -p "$SC/tu/d" "$SC/tu/m" "$SC/out"
rm -f "$SC/out/"*

echo "===INPUTS==="
echo "edited TU (pushed): $(sha256sum "$SC/incoming/k0pf6gm_device_tile_mps.hip" | cut -c1-16) $(wc -l < "$SC/incoming/k0pf6gm_device_tile_mps.hip") lines"
echo "checkout .hip     : $(sha256sum "$FM/k0pf6gm_device_tile_mps.hip" | cut -c1-16)"
echo "vendored p1 body  : $(sha256sum "$FM/n2_phase1_gm_mps.cpp" | cut -c1-16)"
echo "donor p1 body     : $(sha256sum "$K0/solution/hip/n2_phase1_gm.cpp" | cut -c1-16)"

cp "$SC/incoming/k0pf6gm_device_tile_mps.hip" "$SC/tu/m/k0pf6gm_device_tile_mps.hip"
sed 's|#include "n2_phase1_gm_mps.cpp"|#include "n2_phase1_gm.cpp"|' \
  "$SC/tu/m/k0pf6gm_device_tile_mps.hip" > "$SC/tu/d/k0pf6gm_device_tile_mps.hip"
echo "===TU DIFF (must be exactly the one include line)==="
diff "$SC/tu/d/k0pf6gm_device_tile_mps.hip" "$SC/tu/m/k0pf6gm_device_tile_mps.hip"
echo "===SRC_REV / mask literal in the TU==="
grep -n 'K0P6_MPS_SRC_REV [0-9]\|#define N2GM_P1_SCHED_GSCALE' "$SC/tu/m/k0pf6gm_device_tile_mps.hip"

echo "===BUILD (subha_k1, CPU-only genco, 5-way)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e26act
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1
  case $v in
    D) SRC=$SC/tu/d/k0pf6gm_device_tile_mps.hip; EXTRA="" ;;
    *) SRC=$SC/tu/m/k0pf6gm_device_tile_mps.hip; EXTRA="" ;;
  esac
  # masks other than 0 are produced by rewriting the hashed literal, NOT by -D,
  # so the TU that gets measured is byte-for-byte the TU that gets committed.
  if [ "$v" != "D" ] && [ "$v" != "M0" ]; then
    m=${v#M}
    sed "s/#define N2GM_P1_SCHED_GSCALE 0/#define N2GM_P1_SCHED_GSCALE $m/" \
      $SC/tu/m/k0pf6gm_device_tile_mps.hip > $SC/tu/m/tu_$v.hip
    SRC=$SC/tu/m/tu_$v.hip
  fi
  t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $EXTRA "$SRC" -o $SC/out/$v.hsaco > $SC/out/$v.log 2>&1
  rc=$?
  echo "$v exit=$rc secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/$v.log)"
}
export -f build_one; export SC DHK K0 MR
printf "%s\n" D M0 M4 M1 M5 | xargs -P 5 -I{} bash -c "build_one {}"
echo "===UNBUNDLE + .text==="
for v in D M0 M4 M1 M5; do
  [ -f $SC/out/$v.hsaco ] || { echo "$v MISSING HSACO"; continue; }
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/$v.text.bin $SC/out/$v.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/$v.elf > $SC/out/$v.isa 2>/dev/null
done
' 2>&1 | tail -25

echo "===ERRORS (if any)==="
for v in D M0 M4 M1 M5; do
  n=$(grep -cE "error:" "$SC/out/$v.log" 2>/dev/null || echo 0)
  [ "$n" != "0" ] && { echo "---- $v ----"; grep -E "error:" "$SC/out/$v.log" | head -8; }
done

echo "===THE GATE: .text sha256 + size==="
for v in D M0 M4 M1 M5; do
  if [ -f "$SC/out/$v.text.bin" ]; then
    printf '%-4s %s  %s B\n' "$v" "$(sha256sum "$SC/out/$v.text.bin" | cut -d" " -f1)" \
      "$(stat -c %s "$SC/out/$v.text.bin")"
  else
    printf '%-4s (no .text)\n' "$v"
  fi
done
if cmp -s "$SC/out/D.text.bin" "$SC/out/M0.text.bin"; then
  echo "GATE PASS: mask 0 .text is BYTE-IDENTICAL to the donor-include build"
else
  echo "GATE FAIL: mask 0 .text DIFFERS from the donor-include build"
  cmp "$SC/out/D.text.bin" "$SC/out/M0.text.bin" | head -3
fi

echo "===RESOURCE TUPLE (per build)==="
for v in D M0 M4 M1 M5; do
  echo "---- $v ----"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' \
    "$SC/out/$v.log" 2>/dev/null | sed 's/^remark: //' | \
    awk '/k0pf6gm_mps_mega/{p=1} p' | head -14
done

echo "===SCRATCH INSTRUCTION COUNT (whole kernel)==="
for v in D M0 M4 M1 M5; do
  [ -f "$SC/out/$v.isa" ] && printf '%-4s scratch_ops=%s  atomics=%s\n' "$v" \
    "$(grep -cE '^\s+scratch_(load|store)' "$SC/out/$v.isa")" \
    "$(grep -cE 'flat_atomic_pk_add_bf16' "$SC/out/$v.isa")"
done
echo "===DONE==="
exit 0
