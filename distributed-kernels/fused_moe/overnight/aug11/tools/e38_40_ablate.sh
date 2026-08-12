#!/usr/bin/env bash
# exp_38 SITE ABLATION: which of mode 14's nine insertion sites actually moves
# the schedule? CPU-only, one hipcc per site, run SERIALLY so a timing campaign
# on the GPUs is never competing for cores.
#
# Method: rename the Nth `#if K0P6_MPS_ENABLE_MODE14` in each file to a
# per-site macro, then build with exactly one site defined. An undefined macro
# is 0 to the preprocessor, so no default block is needed and the transform is a
# pure rename -- the ONLY difference between a site build and DEF is which one
# `#if` went true.
#
# Site map (file order):
#   S1 mode_is_direct_accum        third mode comparison            (cuh)
#   S2 skip_dead_part_zero         mode-14 admission                (cuh)
#   S3 config_is_valid + GLegalBits validation rules                (cuh)
#   S4 k0p6_mps_task_done          early return for mode 14         (hip)
#   S5 k0p6_mps_task_drain         third mode comparison            (hip)
#   S6 task_done_maybe_defer       packed-word split + mode-14 arm  (hip)
#   S7 kernel entry guard          row_ready tail-index check       (hip)
#   S8 M7.5 rendezvous             cfg75/coarse75 + publish + poll  (hip)
#   S9 M8                          m8_coarse + Ready instantiation  (hip)
set -uo pipefail
SC=$HOME/overnight-scratch/e38
DHK=$HOME/Distributed-HipKittens
IN=$SC/incoming
REF_TEXT=$SC/out/REF.text.bin
SITES="S1 S2 S3 S4 S5 S6 S7 S8 S9"

# occurrence -> site, in file order
HIP_MAP="S4 S5 S6 S6 S7 S8 S8 S8 S8 S8 S9 S9 S9"
CUH_MAP="S3 S1 S2 S3 S3 S3"

[ -f "$REF_TEXT" ] || { echo "no REF .text -- run e38_10_gate.sh first"; exit 2; }

renumber() {  # renumber <infile> <outfile> <map...>
  local in="$1" out="$2"; shift 2
  awk -v map="$*" '
    BEGIN { n = split(map, M, " "); i = 0 }
    /^#if K0P6_MPS_ENABLE_MODE14$/ {
      i++
      if (i <= n) { print "#if K0P6_M14_" M[i]; next }
    }
    { print }
    END { if (i != n) printf("** MAP MISMATCH: saw %d guards, map has %d **\n", i, n) > "/dev/stderr" }
  ' "$in" > "$out"
}

echo "===PREPARE ABLATION TU==="
mkdir -p "$SC/tu/ABL"
renumber "$IN/k0pf6gm_device_tile_mps.hip" "$SC/tu/ABL/k0pf6gm_device_tile_mps.hip" $HIP_MAP
renumber "$IN/moe_mps_adapter.cuh"        "$SC/tu/ABL/moe_mps_adapter.cuh"        $CUH_MAP
echo "  per-site guard census in the ablation TU:"
for s in $SITES; do
  printf '    %-3s hip=%s cuh=%s\n' "$s" \
    "$(grep -c "^#if K0P6_M14_$s\$" "$SC/tu/ABL/k0pf6gm_device_tile_mps.hip")" \
    "$(grep -c "^#if K0P6_M14_$s\$" "$SC/tu/ABL/moe_mps_adapter.cuh")"
done
echo "  residual unrenamed ENABLE_MODE14 guards (must be 0/0): hip=$(grep -c '^#if K0P6_MPS_ENABLE_MODE14$' "$SC/tu/ABL/k0pf6gm_device_tile_mps.hip") cuh=$(grep -c '^#if K0P6_MPS_ENABLE_MODE14$' "$SC/tu/ABL/moe_mps_adapter.cuh")"

echo
echo "===BUILD one site at a time (serial, nice'd)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e38
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
for s in S1 S2 S3 S4 S5 S6 S7 S8 S9 ALL; do
  if [ "$s" = ALL ]; then D="-DK0P6_M14_S1=1 -DK0P6_M14_S2=1 -DK0P6_M14_S3=1 -DK0P6_M14_S4=1 -DK0P6_M14_S5=1 -DK0P6_M14_S6=1 -DK0P6_M14_S7=1 -DK0P6_M14_S8=1 -DK0P6_M14_S9=1"
  else D="-DK0P6_M14_$s=1"; fi
  nice -n 19 hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage $D \
    -I$SC/tu/ABL \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/ABL/k0pf6gm_device_tile_mps.hip -o $SC/out/A_$s.hsaco > $SC/out/A_$s.log 2>&1
  echo "$s exit=$? errors=$(grep -cE "error:" $SC/out/A_$s.log)"
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/A_$s.hsaco --output=$SC/out/A_$s.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/A_$s.text.bin $SC/out/A_$s.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/A_$s.elf > $SC/out/A_$s.isa 2>/dev/null
done
' 2>&1 | sed 's/^/  /'

echo
echo "===RESULT: which single site breaks .text identity with rev 26?========"
printf '  %-4s %-9s %-9s %-8s %-8s %-7s %s\n' site text_eq text_B sgpr_sp vgpr_sp scr_ops verdict
for s in $SITES ALL; do
  t=$SC/out/A_$s.text.bin
  [ -f "$t" ] || { printf '  %-4s (no .text)\n' "$s"; continue; }
  if cmp -s "$REF_TEXT" "$t"; then eq=IDENTICAL; verdict="inert"; else eq=DIFFERS; verdict="PERTURBS"; fi
  sg=$(grep -oE 'SGPRs Spill: [0-9]+' "$SC/out/A_$s.log" | head -1 | grep -oE '[0-9]+')
  vg=$(grep -oE 'VGPRs Spill: [0-9]+' "$SC/out/A_$s.log" | head -1 | grep -oE '[0-9]+')
  so=$(grep -cE '^\s+scratch_(load|store)' "$SC/out/A_$s.isa" 2>/dev/null)
  printf '  %-4s %-9s %-9s %-8s %-8s %-7s %s\n' "$s" "$eq" "$(stat -c %s "$t")" "${sg:-?}" "${vg:-?}" "${so:-?}" "$verdict"
done
echo
echo "  reference: REF/DEF = $(stat -c %s "$REF_TEXT") B, SGPR spill 186, VGPR spill 15, scratch_ops 19"
echo "===DONE==="
exit 0
