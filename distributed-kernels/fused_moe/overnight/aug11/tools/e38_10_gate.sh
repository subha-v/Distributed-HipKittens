#!/usr/bin/env bash
# exp_38 GATE: is the DEFAULT build's .text byte-identical to rev 26 (f113d73f)?
#
# CPU-only, in subha_k1. Four TUs, each in its own directory so the QUOTED
# include `moe_mps_adapter.cuh` resolves TU-locally and the node checkout is
# never touched (the e27_build.sh pattern):
#
#   REF  = f113d73f content of both files                  -- the reference
#   DEF  = exp_38 guarded content, NO -D at all            -- what ships
#   M14  = exp_38 guarded content, -DK0P6_MPS_ENABLE_MODE14=1
#   RING = exp_38 guarded content, -DK0P6_MPS_E23_RING=1
#
# GATE (the only one we trust now): sha256(.text(DEF)) == sha256(.text(REF)).
# M14 and RING must (a) compile clean and (b) DIFFER from REF -- if a flag-on
# build is identical to REF the flag is not reaching the code and the arm is a
# lie. The resource tuple is printed but is explicitly a SECONDARY check: the
# mode-14 commit matched every field of it and still cost 726.9 us.
set -uo pipefail
SC=$HOME/overnight-scratch/e38
DHK=$HOME/Distributed-HipKittens
IN=$SC/incoming
REFCOMMIT=f113d73f
VARIANTS="REF DEF M14 RING"

mkdir -p "$SC/out" "$IN"
for v in $VARIANTS; do mkdir -p "$SC/tu/$v"; done
rm -f "$SC/out/"*

echo "===INPUTS (uploaded working-tree files)==="
for f in k0pf6gm_device_tile_mps.hip moe_mps_adapter.cuh; do
  [ -f "$IN/$f" ] || { echo "MISSING $IN/$f"; exit 2; }
  # Normalize CRLF that a Windows-side scp may have introduced. Line endings
  # cannot change .text, but they can make a `cmp` of SOURCES useless, and the
  # source cmp is how we prove the guard is the only difference.
  tr -d '\r' < "$IN/$f" > "$IN/$f.lf" && mv "$IN/$f.lf" "$IN/$f"
  printf '  %-34s %s  %6s lines\n' "$f" \
    "$(sha256sum "$IN/$f" | cut -c1-16)" "$(wc -l < "$IN/$f")"
done

echo
echo "===REFERENCE EXTRACTION ($REFCOMMIT)==="
git -C "$DHK" cat-file -e "$REFCOMMIT^{commit}" 2>/dev/null || {
  echo "commit $REFCOMMIT not present on the node -- fetching"; git -C "$DHK" fetch --all -q; }
for f in k0pf6gm_device_tile_mps.hip moe_mps_adapter.cuh; do
  git -C "$DHK" show "$REFCOMMIT:distributed-kernels/fused_moe/$f" \
    > "$SC/tu/REF/$f" || { echo "cannot extract $f at $REFCOMMIT"; exit 2; }
  printf '  %-34s %s  %6s lines\n' "$f" \
    "$(sha256sum "$SC/tu/REF/$f" | cut -c1-16)" "$(wc -l < "$SC/tu/REF/$f")"
done
echo "  REF SRC_REV = $(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$SC/tu/REF/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')  (expect 26)"

echo
echo "===LAY OUT DEF / M14 / RING (identical sources; only -D differs)==="
for v in DEF M14 RING; do
  cp "$IN/k0pf6gm_device_tile_mps.hip" "$SC/tu/$v/"
  cp "$IN/moe_mps_adapter.cuh"        "$SC/tu/$v/"
done
echo "  DEF SRC_REV = $(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$SC/tu/DEF/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')  (expect 30)"
echo "  DEF flag defaults:"
grep -nE '^#define (K0P6_MPS_ENABLE_MODE14|K0P6_MPS_E23_RING|K0P6_MPS_E23_FORCE_ON) ' \
  "$SC/tu/DEF/moe_mps_adapter.cuh" | sed 's/^/    /'
echo "  guard-site census in DEF (#if K0P6_MPS_ENABLE_MODE14):"
printf '    hip=%s  cuh=%s\n' \
  "$(grep -c '^#if K0P6_MPS_ENABLE_MODE14' "$SC/tu/DEF/k0pf6gm_device_tile_mps.hip")" \
  "$(grep -c '^#if K0P6_MPS_ENABLE_MODE14' "$SC/tu/DEF/moe_mps_adapter.cuh")"

echo
echo "===BUILD (subha_k1, CPU-only genco, 4-way parallel)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e38
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1; shift
  t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage "$@" \
    -I$SC/tu/$v \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $SC/out/$v.hsaco > $SC/out/$v.log 2>&1
  echo "$v exit=$? secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/$v.log)"
}
export -f build_one; export SC DHK K0 MR
( build_one REF & build_one DEF & \
  build_one M14 -DK0P6_MPS_ENABLE_MODE14=1 & \
  build_one RING -DK0P6_MPS_E23_RING=1 & wait )
echo "===UNBUNDLE + .text==="
for v in REF DEF M14 RING; do
  [ -f $SC/out/$v.hsaco ] || { echo "$v MISSING HSACO"; continue; }
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/$v.text.bin $SC/out/$v.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/$v.elf > $SC/out/$v.isa 2>/dev/null
done
' 2>&1 | sed 's/^/  /'

echo
echo "===COMPILE ERRORS (if any)==="
for v in $VARIANTS; do
  n=$(grep -cE "error:" "$SC/out/$v.log" 2>/dev/null || echo 0)
  if [ "$n" != "0" ]; then echo "---- $v ($n) ----"; grep -E "error:" "$SC/out/$v.log" | head -12; fi
done
echo "(nothing above == all four compiled clean)"

echo
echo "===THE GATE: .text sha256 =========================================="
for v in $VARIANTS; do
  if [ -f "$SC/out/$v.text.bin" ]; then
    printf '  %-4s %s  %s B\n' "$v" \
      "$(sha256sum "$SC/out/$v.text.bin" | cut -d' ' -f1)" \
      "$(stat -c %s "$SC/out/$v.text.bin")"
  else printf '  %-4s (no .text)\n' "$v"; fi
done
echo
if cmp -s "$SC/out/REF.text.bin" "$SC/out/DEF.text.bin"; then
  echo "  GATE PASS: DEF .text is BYTE-IDENTICAL to rev 26. The ratchet is restored."
else
  echo "  GATE FAIL: DEF .text DIFFERS from rev 26."
  echo "    sizes: REF=$(stat -c %s "$SC/out/REF.text.bin" 2>/dev/null) DEF=$(stat -c %s "$SC/out/DEF.text.bin" 2>/dev/null)"
  cmp -l "$SC/out/REF.text.bin" "$SC/out/DEF.text.bin" 2>/dev/null | wc -l | sed 's/^/    differing bytes: /'
  cmp "$SC/out/REF.text.bin" "$SC/out/DEF.text.bin" 2>/dev/null | head -3 | sed 's/^/    /'
fi
for v in M14 RING; do
  if cmp -s "$SC/out/REF.text.bin" "$SC/out/$v.text.bin"; then
    echo "  ** $v .text == REF -- the flag is NOT reaching the code, that arm would be a lie **"
  else
    echo "  OK: $v .text differs from REF (flag-on really changes the binary)"
  fi
done

echo
echo "===SECONDARY (demoted) : resource tuple==============================="
for v in $VARIANTS; do
  echo "---- $v ----"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' \
    "$SC/out/$v.log" 2>/dev/null | sed 's/^remark: //' | \
    awk '/k0pf6gm_mps_mega/{p=1} p' | head -14 | sed 's/^/  /'
done

echo
echo "===ISA census (mfma / scratch / throttle vmcnt instantiations)========="
for v in $VARIANTS; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '  %-4s mfma=%s scratch_ops=%s pk_add_bf16=%s vmcnt4=%s vmcnt8=%s vmcnt16=%s vmcnt32=%s\n' "$v" \
    "$(grep -cE '^\s+v_mfma' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+scratch_(load|store)' "$SC/out/$v.isa")" \
    "$(grep -c 'flat_atomic_pk_add_bf16' "$SC/out/$v.isa")" \
    "$(grep -cE 's_waitcnt vmcnt\(4\)' "$SC/out/$v.isa")" \
    "$(grep -cE 's_waitcnt vmcnt\(8\)' "$SC/out/$v.isa")" \
    "$(grep -cE 's_waitcnt vmcnt\(16\)' "$SC/out/$v.isa")" \
    "$(grep -cE 's_waitcnt vmcnt\(32\)' "$SC/out/$v.isa")"
done
echo "===DONE==="
exit 0
