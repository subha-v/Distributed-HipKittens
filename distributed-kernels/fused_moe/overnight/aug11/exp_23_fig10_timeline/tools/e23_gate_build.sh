#!/usr/bin/env bash
# exp_23 PARITY GATE. CPU-only (hipcc --genco in subha_k1). NO GPU.
#
# Proves the per-CTA phase stamp array is FREE when disabled: the instrumented
# TU must have a resource tuple byte-identical to the arm being published, or the
# traced timeline is not that arm's timeline and Fig 3 is invalid.
#
# Four TUs, each in its OWN directory so the quoted #include resolves TU-locally
# and the node checkout is never touched:
#   R  = the node checkout as-is                       -- the published arm
#   TA = R + exp_23 tier A  (5 marks, all in existing ts_last blocks)
#   TB = TA + tier B        (+3 marks: KSTART, M8_ENTER, M9_DONE)
#   TC = TB + tier C        (+ service / M7.5 / META)
#
# The tiers are separate TUs on purpose: a single all-or-nothing build tells you
# only THAT it failed, and the whole value of the ladder is knowing WHERE.
#
# GATES (see patch_spec.md section 6)
#   G0  every TU compiles clean
#   G1  SGPR 106 / ArchVGPR 256 / AGPR 256 / Scratch 128 B / LDS 155,496 exact
#   G2  Occupancy == 1 wave/SIMD  (asserted, never assumed)
#   G3  v_mfma census == 180 and flat_atomic_pk_add_bf16 == 180-span-safe 282
#   G4  zero scratch ops inside either MFMA span
#   G5  SGPR spills <= 186, VGPR spills <= 14
#   G6  .text(TA) != .text(R)   -- the instrument really compiled in
#   G7  s_memrealtime count: TA must add ZERO vs R (proves ts_mark shared the
#       clock read instead of doing ts_last(); e23_mark())
#
# USAGE
#   1. Put the four .hip variants in $SC/incoming/ as
#      ratchet_k0pf6gm_device_tile_mps.hip, tierA_..., tierB_..., tierC_...
#      plus ratchet_moe_mps_adapter.cuh and tier_moe_mps_adapter.cuh
#      (the adapter is additive, so ONE instrumented copy serves all three tiers)
#   2. bash e23_gate_build.sh
#   3. Read the G-lines. Ship the highest tier that is green on G1-G5.
set -uo pipefail
SC=$HOME/overnight-scratch/e23
DHK=$HOME/Distributed-HipKittens
FM=$DHK/distributed-kernels/fused_moe
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
IN=$SC/incoming
VARIANTS="R TA TB TC"
# The reference tuple. STATUS.md:145 (exp_34, mode 14 in the build). NOT the
# stale SGPR 104 / LDS 155,428 in CLAUDE.md.
REF_SGPR=106; REF_VGPR=256; REF_AGPR=256; REF_SCRATCH=128; REF_LDS=155496
REF_MFMA=180; REF_PKADD=282; REF_OCC=1

mkdir -p "$SC/out" $(for v in $VARIANTS; do echo "$SC/tu/$v"; done)
rm -f "$SC/out/"*

echo "===INPUTS==="
for f in ratchet_k0pf6gm_device_tile_mps.hip ratchet_moe_mps_adapter.cuh \
         tierA_k0pf6gm_device_tile_mps.hip tierB_k0pf6gm_device_tile_mps.hip \
         tierC_k0pf6gm_device_tile_mps.hip tier_moe_mps_adapter.cuh; do
  [ -f "$IN/$f" ] || { echo "MISSING $IN/$f"; exit 2; }
  printf '%-46s %s  %s lines\n' "$f" "$(sha256sum "$IN/$f" | cut -c1-16)" "$(wc -l < "$IN/$f")"
done
echo "node checkout .hip:     $(sha256sum "$FM/k0pf6gm_device_tile_mps.hip" | cut -c1-16)"
echo "node checkout adapter:  $(sha256sum "$FM/moe_mps_adapter.cuh" | cut -c1-16)"

echo
echo "===LAY OUT THE TUs==="
cp "$IN/ratchet_k0pf6gm_device_tile_mps.hip" "$SC/tu/R/k0pf6gm_device_tile_mps.hip"
cp "$IN/ratchet_moe_mps_adapter.cuh"         "$SC/tu/R/moe_mps_adapter.cuh"
# The reference MUST equal the node checkout or the whole gate is against a
# phantom. This is the check exp_27's gate script earned the hard way.
cmp -s "$SC/tu/R/k0pf6gm_device_tile_mps.hip" "$FM/k0pf6gm_device_tile_mps.hip" \
  && echo "R .hip     == node checkout  OK" || echo "R .hip     != node checkout  ** FATAL **"
cmp -s "$SC/tu/R/moe_mps_adapter.cuh" "$FM/moe_mps_adapter.cuh" \
  && echo "R adapter  == node checkout  OK" || echo "R adapter  != node checkout  ** FATAL **"
for v in TA TB TC; do
  cp "$IN/tier${v#T}_k0pf6gm_device_tile_mps.hip" "$SC/tu/$v/k0pf6gm_device_tile_mps.hip"
  cp "$IN/tier_moe_mps_adapter.cuh"               "$SC/tu/$v/moe_mps_adapter.cuh"
done
echo "--- SRC_REV per TU (every instrumented TU must be > R's) ---"
for v in $VARIANTS; do
  printf '%-3s SRC_REV=%s\n' "$v" \
    "$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')"
done
echo "--- mark-site census per TU (sanity: A=5, B=8, C=14) ---"
for v in $VARIANTS; do
  printf '%-3s ts_mark=%s e23_mark=%s e23_meta=%s ts_last=%s\n' "$v" \
    "$(grep -c 'ts_mark(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'e23_mark(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'e23_meta(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c 'ts_last(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip")"
done

echo
echo "===BUILD (subha_k1, CPU-only genco, 4-way)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e23
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1; t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    -I$SC/tu/$v \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $SC/out/$v.hsaco > $SC/out/$v.log 2>&1
  echo "$v exit=$? secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/$v.log)"
}
export -f build_one; export SC DHK K0 MR
printf "%s\n" R TA TB TC | xargs -P 4 -I{} bash -c "build_one {}"
for v in R TA TB TC; do
  [ -f $SC/out/$v.hsaco ] || { echo "$v MISSING HSACO"; continue; }
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/$v.text.bin $SC/out/$v.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/$v.elf > $SC/out/$v.isa 2>/dev/null
done
' 2>&1 | tail -24

echo
echo "===G0: compile errors==="
for v in $VARIANTS; do
  n=$(grep -cE "error:" "$SC/out/$v.log" 2>/dev/null || echo 0)
  [ "$n" = "0" ] || { echo "---- $v ($n) ----"; grep -E "error:" "$SC/out/$v.log" | head -12; }
done
echo "(no output above == all four compiled clean)"

echo
echo "===G1/G2/G5: resource tuple + occupancy + spills (k0pf6gm_mps_mega)==="
for v in $VARIANTS; do
  echo "---- $v ----"
  awk '/k0pf6gm_mps_mega/{p=1} p' "$SC/out/$v.log" 2>/dev/null \
    | grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' \
    | sed 's/^remark: //' | head -14
done
echo
echo "--- G1/G2 verdict (exact match required against R and against the reference) ---"
getf() { awk '/k0pf6gm_mps_mega/{p=1} p' "$SC/out/$1.log" 2>/dev/null \
  | grep -m1 -E "$2" | grep -oE '[0-9]+' | tail -1; }
for v in $VARIANTS; do
  s=$(getf $v 'SGPRs'); a=$(getf $v 'AGPRs'); g=$(getf $v 'ScratchSize')
  l=$(getf $v 'LDS Size'); o=$(getf $v 'Occupancy')
  # ArchVGPRs and AGPRs are printed on separate lines; VGPRs is the arch count.
  vv=$(awk '/k0pf6gm_mps_mega/{p=1} p' "$SC/out/$v.log" | grep -m1 -E 'VGPRs' | grep -oE '[0-9]+' | tail -1)
  ok=PASS
  [ "$s" = "$REF_SGPR" ]    || ok="FAIL(SGPR=$s want $REF_SGPR)"
  [ "$vv" = "$REF_VGPR" ]   || ok="$ok FAIL(VGPR=$vv)"
  [ "$a" = "$REF_AGPR" ]    || ok="$ok FAIL(AGPR=$a)"
  [ "$g" = "$REF_SCRATCH" ] || ok="$ok FAIL(scratch=$g want $REF_SCRATCH)"
  [ "$l" = "$REF_LDS" ]     || ok="$ok FAIL(LDS=$l want $REF_LDS)"
  [ "$o" = "$REF_OCC" ]     || ok="$ok FAIL(occupancy=$o want $REF_OCC -- ASSERTED, NOT ASSUMED)"
  printf 'G1/G2 %-3s %s\n' "$v" "$ok"
done

echo
echo "===G3/G4/G7: census, scratch ops, clock reads==="
printf '%-3s %-10s %-14s %-13s %-14s %s\n' TU v_mfma pk_add_bf16 scratch_ops s_memrealtime s_barrier
for v in $VARIANTS; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '%-3s %-10s %-14s %-13s %-14s %s\n' "$v" \
    "$(grep -cE '^\s+v_mfma' "$SC/out/$v.isa")" \
    "$(grep -c 'flat_atomic_pk_add_bf16' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+scratch_(load|store)' "$SC/out/$v.isa")" \
    "$(grep -c 's_memrealtime' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+s_barrier' "$SC/out/$v.isa")"
done
echo "G3 requires v_mfma == $REF_MFMA and pk_add_bf16 == $REF_PKADD on EVERY TU."
echo "G7 requires s_memrealtime(TA) == s_memrealtime(R): tier A must add ZERO clock"
echo "   reads. A rise means ts_last(); e23_mark() was written instead of ts_mark()."
echo "G4: scratch op COUNT may move, but none may land inside either MFMA span."
echo "    Span check (the standing rule) -- longest v_mfma run with no scratch op:"
for v in $VARIANTS; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '  %-3s ' "$v"
  awk '/v_mfma/{m++; if(m==1)s=NR} /scratch_(load|store)/{if(m>0){print "SCRATCH OP INSIDE AN MFMA REGION at line " NR; bad=1; exit}} END{if(!bad) print "clean (" m " mfma, 0 interleaved scratch)"}' \
    "$SC/out/$v.isa"
done

echo
echo "===G6: .text sha256 (TA..TC must all DIFFER from R)==="
for v in $VARIANTS; do
  if [ -f "$SC/out/$v.text.bin" ]; then
    printf '%-3s %s  %s B\n' "$v" "$(sha256sum "$SC/out/$v.text.bin" | cut -c1-32)" \
      "$(stat -c %s "$SC/out/$v.text.bin")"
  else printf '%-3s (no .text)\n' "$v"; fi
done
for v in TA TB TC; do
  if cmp -s "$SC/out/R.text.bin" "$SC/out/$v.text.bin"; then
    echo "G6 $v ** FAIL: .text identical to R -- the instrument did NOT compile in **"
  else
    echo "G6 $v PASS: .text differs from R"
  fi
done

echo
echo "===SHIP DECISION==="
echo "Ship the HIGHEST tier that is PASS on G1, G2, G3, G4, G5 and G7."
echo "If TA itself fails, do not ship a timeline: fall back to exp_33's coarse"
echo "stamps and record Q4b as NOT STARTED with the parity failure as the reason."
echo "===DONE==="
exit 0
