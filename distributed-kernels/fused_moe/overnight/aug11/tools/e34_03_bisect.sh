#!/usr/bin/env bash
# exp_34 step 3: attribute the scratch regression (128 -> 144 B/lane, SGPR spill
# 186 -> 221, VGPR spill 15 -> 21) to a specific piece of the patch.
# Build-only probes, NOT shippable arms. NO GPU WORK.
#
#   A  = pristine ca5b683f                                (128 / 186 / 15)
#   B  = full exp_34 patch                                (144 / 221 / 21)
#   P1 = B with the Ready=true M8 instantiation REMOVED (mode 14 shares mode 12's
#        M8 body). Isolates "a fifth inlined combine body".
#   P2 = B with the M7.7 rendezvous block compiled out. Isolates the new phase.
#   P3 = B with both removed  -> should return to A if those two are the whole
#        story; anything left over is the per-task hook / plumbing edits.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
FMK=$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
mkdir -p "$SC/tu"

python3 - "$FMK" "$SC/tu" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])

READY_INST = "false, true, true>"
assert src.count(READY_INST) == 1, src.count(READY_INST)
COARSE_GUARD = "if (hk_moe::mps::mode_is_coarse(ccfg)) {"
assert src.count(COARSE_GUARD) == 1, src.count(COARSE_GUARD)

p1 = src.replace(READY_INST, "false, true>")
p2 = src.replace(COARSE_GUARD, "if (false && hk_moe::mps::mode_is_coarse(ccfg)) {")
p3 = p1.replace(COARSE_GUARD, "if (false && hk_moe::mps::mode_is_coarse(ccfg)) {")
for name, text in (("P1", p1), ("P2", p2), ("P3", p3)):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "k0pf6gm_device_tile_mps.hip").write_text(text)
    print(name, "written", len(text))
PY

docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla; SC=$H/e34; O=$SC/out
DHK=$SC/DHK; FM=$DHK/distributed-kernels/fused_moe
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
COMMON=(--genco --offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in P1 P2 P3; do
  echo "### $v"
  hipcc "${COMMON[@]}" $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco \
    > $O/$v.log 2>&1
  echo "$v exit=$? errors=$(grep -cE "error:" $O/$v.log)"
done
'

echo
echo "############ TUPLES ############"
printf '%-4s %-8s %-6s %-6s %-8s %-10s %-10s %-8s\n' arm SGPR VGPR AGPR scratch sgpr_spill vgpr_spill LDS
for v in A B P1 P2 P3; do
  f=$O/$v.log
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  printf '%-4s %-8s %-6s %-6s %-8s %-10s %-10s %-8s\n' "$v" \
    "$(g 'TotalSGPRs')" "$(g '^.*VGPRs:')" "$(g 'AGPRs:')" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" "$(g 'LDS Size')"
done

echo
echo "############ ISA CENSUS ############"
docker exec subha_k1 bash -lc '
SC=/home/subvadla/e34; O=$SC/out
for v in A B P1 P2 P3; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $O/$v.hsaco > $O/$v.isa 2>&1
done
echo objdump_done'
printf '%-4s %-10s %-10s %-12s %-10s %-10s\n' arm isa_lines v_mfma pk_add_bf16 scratch buffer_inv
for v in A B P1 P2 P3; do
  printf '%-4s %-10s %-10s %-12s %-10s %-10s\n' "$v" \
    "$(wc -l < $O/$v.isa)" \
    "$(grep -c 'v_mfma' $O/$v.isa)" \
    "$(grep -c 'flat_atomic_pk_add_bf16' $O/$v.isa)" \
    "$(grep -c 'scratch_' $O/$v.isa)" \
    "$(grep -c 'buffer_inv' $O/$v.isa)"
done
echo "===DONE==="
