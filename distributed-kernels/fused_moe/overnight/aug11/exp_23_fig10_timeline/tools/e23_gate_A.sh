#!/usr/bin/env bash
# exp_23 TIER A PARITY GATE. CPU-only (hipcc --genco in subha_k1). NO GPU.
#
# Question this answers: does the per-CTA phase ring, COMPILED IN BUT RUNTIME
# OFF, cost anything? If it does, the traced timeline is not the timeline of
# the arm we publish and Fig 3 is invalid.
#
# Three builds from two TU directories:
#   R     scratch clone at the published HEAD           -- the arm
#   TA    R + tier A (5 ts_mark swaps + adapter block)  -- THE SHIPPED BINARY
#   TAON  TA source, -DK0P6_MPS_E23_FORCE_ON=1          -- `enable` folded to a
#         compile-time true so the compiler cannot sink the stamps behind a
#         branch. This is the "ring on" tuple. It is a BOUND, not an arm: the
#         shipped binary is TA and collection is turned on by an env flag at
#         runtime, which cannot change a resource tuple.
#   TCON  TA source, FORCE_ON=1 and RING=0               -- coarse stamps folded
#         on with the ring compiled OUT. This exists purely for ATTRIBUTION:
#         TAON alone conflates the ring with the pre-existing coarse
#         atomicMax cells, which also become unconditional. So
#           TCON - R    = cost of `timestamps=1` (already known to be ~15 us
#                         end-to-end; it is NOT the ring's cost)
#           TAON - TCON = cost attributable to the ring itself.
#
# GATES
#   G0  every build compiles clean
#   G1  SGPR/VGPR/AGPR/scratch/LDS exactly equal to the reference tuple
#   G2  Occupancy == 1 wave/SIMD -- ASSERTED, never assumed (a sibling
#       experiment watched an LDS change move pointer arrays to scratch and
#       silently let occupancy rise to 2)
#   G3  v_mfma census and flat_atomic_pk_add_bf16 census unchanged
#   G4  zero scratch ops inside either MFMA span
#   G5  SGPR/VGPR spill counts unchanged vs R
#   G6  .text(TA) != .text(R) -- the instrument really did compile in
#   G7  the fused-clock gate, in two mechanical parts:
#         G7a source census: ts_mark == 5, ts_last == 0, e23_mark == 0 in the
#             instrumented .hip (a bare e23_mark at a stamp site is the
#             substitution we are guarding against)
#         G7b ISA census: s_memrealtime(TA) == s_memrealtime(R), exactly.
#             ts_last(); e23_mark() reads the clock twice and this rises.
set -uo pipefail
SC=$HOME/overnight-scratch/e23
DHK=$HOME/e23/DHK                     # OUR scratch clone; the pinned repo at
                                      # $HOME/Distributed-HipKittens is held by
                                      # the running mode-14 campaign -- do not
                                      # read the arm from it and never reset it.
FM=$DHK/distributed-kernels/fused_moe
IN=$SC/incoming
# Reference tuple: STATUS.md:145 (exp_34, mode 14 in the build). NOT the stale
# SGPR 104 / LDS 155,428 in CLAUDE.md.
REF_SGPR=106; REF_VGPR=256; REF_AGPR=256; REF_SCRATCH=128; REF_LDS=155496
REF_MFMA=180; REF_PKADD=282; REF_OCC=1

mkdir -p "$SC/out" "$SC/tu/R" "$SC/tu/TA"
rm -f "$SC/out/"*

echo "===INPUTS==="
for f in tierA_k0pf6gm_device_tile_mps.hip tier_moe_mps_adapter.cuh; do
  [ -f "$IN/$f" ] || { echo "MISSING $IN/$f"; exit 2; }
  sed -i 's/\r$//' "$IN/$f"
  printf '%-42s %s  %s lines\n' "$f" "$(sha256sum "$IN/$f" | cut -c1-16)" "$(wc -l < "$IN/$f")"
done
echo "scratch clone HEAD: $(git -C "$DHK" rev-parse --short HEAD)  $(git -C "$DHK" status --porcelain | wc -l) dirty files"
printf '%-42s %s\n' "R .hip     (scratch clone)" "$(sha256sum "$FM/k0pf6gm_device_tile_mps.hip" | cut -c1-16)"
printf '%-42s %s\n' "R adapter  (scratch clone)" "$(sha256sum "$FM/moe_mps_adapter.cuh" | cut -c1-16)"

echo
echo "===LAY OUT THE TUs==="
cp "$FM/k0pf6gm_device_tile_mps.hip" "$SC/tu/R/k0pf6gm_device_tile_mps.hip"
cp "$FM/moe_mps_adapter.cuh"         "$SC/tu/R/moe_mps_adapter.cuh"
cp "$IN/tierA_k0pf6gm_device_tile_mps.hip" "$SC/tu/TA/k0pf6gm_device_tile_mps.hip"
cp "$IN/tier_moe_mps_adapter.cuh"          "$SC/tu/TA/moe_mps_adapter.cuh"
echo "--- TA vs R source diff (must be ONLY the SRC_REV bump + 5 stamp sites) ---"
diff -u "$SC/tu/R/k0pf6gm_device_tile_mps.hip" "$SC/tu/TA/k0pf6gm_device_tile_mps.hip" \
  | grep -E '^@@' | sed 's/^/  hip     /'
diff -u "$SC/tu/R/moe_mps_adapter.cuh" "$SC/tu/TA/moe_mps_adapter.cuh" \
  | grep -E '^@@' | sed 's/^/  adapter /'
echo "--- SRC_REV ---"
for v in R TA; do
  printf '  %-4s SRC_REV=%s\n' "$v" \
    "$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')"
done

echo
echo "===G7a: SOURCE census (mechanical, not eyeball)==="
printf '%-4s %-10s %-11s %-11s %s\n' TU ts_mark e23_mark e23_meta ts_last
for v in R TA; do
  printf '%-4s %-10s %-11s %-11s %s\n' "$v" \
    "$(grep -c 'ts_mark(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'e23_mark(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'e23_meta(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'ts_last(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")"
done
A_MARK=$(grep -c 'ts_mark('  "$SC/tu/TA/k0pf6gm_device_tile_mps.hip")
A_E23=$(grep -c 'e23_mark(' "$SC/tu/TA/k0pf6gm_device_tile_mps.hip")
A_LAST=$(grep -c 'ts_last(' "$SC/tu/TA/k0pf6gm_device_tile_mps.hip")
g7a=PASS
[ "$A_MARK" = "5" ]  || g7a="FAIL(ts_mark=$A_MARK want 5)"
[ "$A_E23"  = "0" ]  || g7a="$g7a FAIL(bare e23_mark=$A_E23 at a stamp site -- THE SUBSTITUTION)"
[ "$A_LAST" = "0" ]  || g7a="$g7a FAIL(ts_last=$A_LAST survives in the kernel)"
echo "G7a $g7a"

echo
echo "===BUILD (subha_k1, CPU-only genco)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e23
DHK=$H/e23/DHK
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  name=$1; tu=$2; extra=${3:-}; t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 $extra \
    -Rpass-analysis=kernel-resource-usage \
    -I$SC/tu/$tu \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/$tu/k0pf6gm_device_tile_mps.hip -o $SC/out/$name.hsaco \
    > $SC/out/$name.log 2>&1
  echo "$name exit=$? secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/$name.log)"
}
export -f build_one; export SC DHK K0 MR
( build_one R    R  ""                            ) &
( build_one TA   TA ""                            ) &
( build_one TAON TA "-DK0P6_MPS_E23_FORCE_ON=1"   ) &
( build_one TCON TA "-DK0P6_MPS_E23_FORCE_ON=1 -DK0P6_MPS_E23_RING=0" ) &
wait
for v in R TA TAON TCON; do
  [ -f $SC/out/$v.hsaco ] || { echo "$v MISSING HSACO"; continue; }
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/$v.text.bin $SC/out/$v.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/$v.elf > $SC/out/$v.isa 2>/dev/null
done
' 2>&1 | tail -12

echo
echo "===G0: compile errors==="
for v in R TA TAON TCON; do
  n=$(grep -cE "error:" "$SC/out/$v.log" 2>/dev/null || echo 0)
  [ "$n" = "0" ] || { echo "---- $v ($n) ----"; grep -E "error:" "$SC/out/$v.log" | head -12; }
done
echo "(no output above == all three compiled clean)"

echo
echo "===G1/G2/G5: resource tuple, occupancy, spills (k0pf6gm_mps_mega)==="
for v in R TA TAON TCON; do
  echo "---- $v ----"
  awk '/k0pf6gm_mps_mega/{p=1} p' "$SC/out/$v.log" 2>/dev/null \
    | grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' \
    | sed 's/^remark: //;s/^ *//' | head -14
done
echo
echo "--- G1/G2 verdict: exact equality with the reference tuple ---"
getf() { awk '/k0pf6gm_mps_mega/{p=1} p' "$SC/out/$1.log" 2>/dev/null \
  | grep -m1 -E "$2" | grep -oE '[0-9]+' | tail -1; }
for v in R TA TAON TCON; do
  s=$(getf $v 'SGPRs'); a=$(getf $v 'AGPRs'); g=$(getf $v 'ScratchSize')
  l=$(getf $v 'LDS Size'); o=$(getf $v 'Occupancy')
  vv=$(getf $v 'VGPRs')
  ok=""
  [ "$s" = "$REF_SGPR" ]    || ok="$ok FAIL(SGPR=$s want $REF_SGPR)"
  [ "$vv" = "$REF_VGPR" ]   || ok="$ok FAIL(VGPR=$vv want $REF_VGPR)"
  [ "$a" = "$REF_AGPR" ]    || ok="$ok FAIL(AGPR=$a want $REF_AGPR)"
  [ "$g" = "$REF_SCRATCH" ] || ok="$ok FAIL(scratch=$g want $REF_SCRATCH)"
  [ "$l" = "$REF_LDS" ]     || ok="$ok FAIL(LDS=$l want $REF_LDS)"
  [ "$o" = "$REF_OCC" ]     || ok="$ok FAIL(occupancy=$o want $REF_OCC -- ASSERTED)"
  printf 'G1/G2 %-4s %s\n' "$v" "${ok:-PASS}"
done

echo
echo "===G3/G4/G7b: ISA census==="
printf '%-4s %-9s %-13s %-12s %-14s %s\n' TU v_mfma pk_add_bf16 scratch_ops s_memrealtime s_barrier
for v in R TA TAON TCON; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '%-4s %-9s %-13s %-12s %-14s %s\n' "$v" \
    "$(grep -cE '^\s+v_mfma' "$SC/out/$v.isa")" \
    "$(grep -c 'flat_atomic_pk_add_bf16' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+scratch_(load|store)' "$SC/out/$v.isa")" \
    "$(grep -c 's_memrealtime' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+s_barrier' "$SC/out/$v.isa")"
done
CR=$(grep -c 's_memrealtime' "$SC/out/R.isa" 2>/dev/null || echo -1)
CA=$(grep -c 's_memrealtime' "$SC/out/TA.isa" 2>/dev/null || echo -2)
if [ "$CR" = "$CA" ]; then
  echo "G7b PASS: s_memrealtime R=$CR TA=$CA -- tier A added ZERO clock reads,"
  echo "     so one read feeds both the coarse cell and the ring cell."
else
  echo "G7b ** FAIL **: s_memrealtime R=$CR TA=$CA (delta $((CA-CR)))."
  echo "     A rise means ts_last(); e23_mark() was written instead of ts_mark(),"
  echo "     and the coarse/per-CTA reconciliation is an approximation, not an identity."
fi
echo "G4 scratch ops INSIDE either MFMA span (must be 0 on every TU)."
# Span definition is the established one (tools/e34_42_gate1.sh): group the
# v_mfma line indices into runs whose gaps are < 400 disassembly lines, then
# count scratch traffic strictly inside each run. A naive "any scratch op after
# the first v_mfma" test is NOT this check -- it trips on the reference arm,
# because the two MFMA spans are separated by ordinary spill-carrying code.
for v in R TA TAON TCON; do
  [ -f "$SC/out/$v.isa" ] || continue
  echo "  ---- $v ----"
  python3 - "$SC/out/$v.isa" <<'PY'
import re, sys
lines = open(sys.argv[1]).readlines()
idx = [i for i, l in enumerate(lines) if "v_mfma" in l]
spans, start, prev = [], idx[0], idx[0]
for i in idx[1:]:
    if i - prev > 400:
        spans.append((start, prev)); start = i
    prev = i
spans.append((start, prev))
bad = 0
for a, b in spans:
    n = sum(1 for l in lines[a:b+1] if re.search(r"scratch_(load|store)", l))
    m = sum(1 for j in idx if a <= j <= b)
    bad += n
    print("  span %6d-%-6d  %3d mfma  scratch ops inside = %d" % (a, b, m, n))
print("  G4 %s (%d span(s), %d total scratch ops inside MFMA)"
      % ("PASS" if bad == 0 else "** FAIL **", len(spans), bad))
PY
done

echo
echo "===G6: .text sha256 (TA must DIFFER from R)==="
for v in R TA TAON TCON; do
  if [ -f "$SC/out/$v.text.bin" ]; then
    printf '%-4s %s  %s B\n' "$v" "$(sha256sum "$SC/out/$v.text.bin" | cut -c1-32)" \
      "$(stat -c %s "$SC/out/$v.text.bin")"
  else printf '%-4s (no .text)\n' "$v"; fi
done
for v in TA TAON TCON; do
  if cmp -s "$SC/out/R.text.bin" "$SC/out/$v.text.bin"; then
    echo "G6 $v ** FAIL: .text identical to R -- the instrument did NOT compile in **"
  else
    echo "G6 $v PASS: .text differs from R"
  fi
done
echo "===DONE==="
exit 0
