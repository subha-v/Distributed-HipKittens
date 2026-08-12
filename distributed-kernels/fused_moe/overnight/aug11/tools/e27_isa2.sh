#!/usr/bin/env bash
# exp_27 G5, second attempt. The whole-kernel static counts cannot show a TRIP
# COUNT, and the gather is a loop, so build both arms with line tables and pull
# out exactly the instructions attributed to the gather source lines.
#
#   arm 0 gather : n2_phase1_gm_mps.cpp lines 391-399  (donor, group-major)
#   arm 1 gather : n2_phase1_gm_mps.cpp lines 374-389  (token-major, float4)
#
# -gline-tables-only is used ONLY for attribution; the gated .text is the
# no-debug build from e27_build.sh. Both are reported so any codegen shift from
# the debug flag is visible rather than assumed away.
set -uo pipefail
SC=$HOME/overnight-scratch/e27
O=$SC/out

echo "===BUILD WITH LINE TABLES (A0g, A1g)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e27
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  a=$1
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 -gline-tables-only \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -I$SC/tu/$a \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/$a/k0pf6gm_device_tile_mps.hip -o $SC/out/${a}g.hsaco > $SC/out/${a}g.log 2>&1
  echo "${a}g exit=$? errors=$(grep -cE "error:" $SC/out/${a}g.log)"
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/${a}g.hsaco --output=$SC/out/${a}g.elf 2>/dev/null
  llvm-objdump -d -l --mcpu=gfx950 $SC/out/${a}g.elf > $SC/out/${a}g.isa 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/${a}g.text.bin $SC/out/${a}g.elf /dev/null 2>/dev/null
}
export -f build_one; export SC DHK K0 MR
printf "%s\n" A0 A1 | xargs -P 2 -I{} bash -c "build_one {}"
' 2>&1 | tail -8

echo
echo "===debug-build .text vs gated .text (is the debug build the same code?)==="
for a in A0 A1; do
  printf '%-3s nodebug=%s  debug=%s  %s\n' "$a" \
    "$(sha256sum "$O/$a.text.bin" | cut -c1-16)" \
    "$(sha256sum "$O/${a}g.text.bin" 2>/dev/null | cut -c1-16)" \
    "$(cmp -s "$O/$a.text.bin" "$O/${a}g.text.bin" && echo IDENTICAL || echo differs)"
done

echo
echo "===THE GATHER, ARM 0 (attributed to n2_phase1_gm_mps.cpp:391-400)==="
awk '
  /^; .*n2_phase1_gm_mps\.cpp:/ { split($0,p,":"); ln=p[length(p)]+0; cur=ln }
  { if (cur>=391 && cur<=400) print }
' "$O/A0g.isa" | head -70

echo
echo "===THE GATHER, ARM 1 (attributed to n2_phase1_gm_mps.cpp:374-389)==="
awk '
  /^; .*n2_phase1_gm_mps\.cpp:/ { split($0,p,":"); ln=p[length(p)]+0; cur=ln }
  { if (cur>=374 && cur<=389) print }
' "$O/A1g.isa" | head -90

echo
echo "===INSTRUCTION COUNTS INSIDE THE GATHER REGION, per arm==="
for spec in "A0g 391 400" "A1g 374 389"; do
  set -- $spec
  a=$1; lo=$2; hi=$3
  echo "-- $a lines $lo-$hi --"
  awk -v LO="$lo" -v HI="$hi" '
    /^; .*n2_phase1_gm_mps\.cpp:/ { split($0,p,":"); cur=p[length(p)]+0 }
    /^[[:space:]]+[a-z]/ { if (cur>=LO && cur<=HI) { n++; split($0,f," "); c[f[1]]++ } }
    END { print "  total instructions: " n; for (k in c) printf "  %-28s %d\n", k, c[k] }
  ' "$O/$a.isa" | sort -k2 -rn | head -25
done

echo
echo "===LOOP BOUND EVIDENCE: the compare against the trip limit==="
echo "arm 0 expects a limit of kKGroups*kMrows = 56*96 = 5376 = 0x1500"
echo "arm 1 expects a limit of kScaleQuads*kMrows = 14*96 = 1344 = 0x540"
for a in A0g A1g; do
  echo "-- $a: immediates 0x1500 / 5376 / 0x540 / 1344 / 0xe0 / 224 anywhere --"
  grep -nE '0x1500|, 5376|0x540|, 1344|0xe0[^0-9a-f]|, 224[^0-9]' "$O/$a.isa" | head -12
done
echo "===DONE==="
