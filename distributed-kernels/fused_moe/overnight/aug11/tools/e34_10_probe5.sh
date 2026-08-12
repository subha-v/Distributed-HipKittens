#!/usr/bin/env bash
# exp_34 step 10: the per-task hooks are the suspect for the 96-atomic scratch
# migration -- they are the only edited code INSIDE the phase-2 task path, and
# their mode word arrives through a volatile descriptor read, which is exactly the
# "cannot be proven uniform -> parked in a VGPR -> vector live range across the
# epilogue" defect n2_phase2_gm_mps.cpp:380 fixes with readfirstlane.
#   X1 = N8 + readfirstlane on `m` in k0p6_mps_task_drain and _maybe_defer
#   X2 = N8 with the three per-task hook edits REVERTED (attribution probe only)
#   X3 = X1 + readfirstlane on `k0p6_m` in k0p6_mps_task_done as well
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out

python3 - "$SC" <<'PY'
import sys, pathlib
sc = pathlib.Path(sys.argv[1])
n8 = (sc / "tu/N8/k0pf6gm_device_tile_mps.hip").read_text()

DECL = """  const unsigned long long m =
      ((unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >> 16 &
       0xFFull);
"""
RFL = """  const unsigned long long m =
      (unsigned long long)(unsigned int)__builtin_amdgcn_readfirstlane(
          (int)((unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >>
                16 & 0xFFull));
"""
assert n8.count(DECL) == 2, n8.count(DECL)
x1 = n8.replace(DECL, RFL)

DONE_DECL = """  const unsigned long long k0p6_m =
      (unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >> 16 & 0xFFull;
"""
DONE_RFL = """  const unsigned long long k0p6_m =
      (unsigned long long)(unsigned int)__builtin_amdgcn_readfirstlane(
          (int)((unsigned long long)k0p6_dread(k0p6_desc, K0P6_D_MPS_CFG) >>
                16 & 0xFFull));
"""
assert n8.count(DONE_DECL) == 1
x3 = x1.replace(DONE_DECL, DONE_RFL)

# X2: revert the three per-task hook edits
x2 = n8
x2 = x2.replace("""  if (k0p6_m == 14ull) return;
""", "")
x2 = x2.replace("if (m != 12ull && m != 13ull && m != 14ull) {",
                "if (m != 12ull && m != 13ull) {")
x2 = x2.replace("""  if (m == 14ull) return;
  if (m == 12ull || m == 13ull) {""", "  if (m == 12ull || m == 13ull) {")
assert x2 != n8

for name, text in (("X1", x1), ("X2", x2), ("X3", x3)):
    d = sc / "tu" / name
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
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in X1 X2 X3; do
  hipcc "${COMMON[@]}" --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  echo "$v exit=$? errors=$(grep -cE "error:" $O/$v.log)"
  hipcc "${COMMON[@]}" --cuda-device-only -S $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
done
echo asm_done'

echo
printf '%-4s %-8s %-11s %-11s %-9s %-8s %-8s\n' arm scratch sgpr_spill vgpr_spill scr_load mfma pk_add
for v in A W2 N8 X1 X2 X3; do
  f=$O/$v.log; s=$O/$v.s
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  printf '%-4s %-8s %-11s %-11s %-9s %-8s %-8s\n' "$v" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" \
    "$(c scratch_load)" "$(c v_mfma)" "$(c flat_atomic_pk_add_bf16)"
done
echo
echo "############ exp_26 SIGNATURE ############"
for v in A W2 N8 X1 X2 X3; do
  [ -s "$O/$v.s" ] || continue
  printf '%-4s ' "$v"
  awk '
    /^[[:space:]]+v_mfma/                  { ++nm; mline[nm]=NR }
    /^[[:space:]]+flat_atomic_pk_add_bf16/ { ++na; aline[na]=NR }
    /^[[:space:]]+scratch_(load|store)/    { ++ns; sline[ns]=NR }
    END {
      sp=0; start=mline[1]; prev=mline[1]
      for (i=2;i<=nm;i++) { if (mline[i]-prev>2000) { sp++; lo[sp]=start; hi[sp]=prev; start=mline[i] } prev=mline[i] }
      sp++; lo[sp]=start; hi[sp]=prev
      inm=0
      for (j=1;j<=sp;j++) for (i=1;i<=ns;i++) if (sline[i]>=lo[j] && sline[i]<=hi[j]) inm++
      h=0
      for (i=1;i<=na;i++) for (j=1;j<=ns;j++) if (sline[j]<aline[i] && aline[i]-sline[j]<=40) { h++; break }
      btw=0
      for (i=1;i<=ns;i++) if (sline[i]>=aline[1] && sline[i]<=aline[na]) btw++
      printf "in_mfma=%d  between_atomics=%d  atomics_with_scratch_ahead=%d/%d\n", inm, btw, h, na
    }' "$O/$v.s"
done
echo "===DONE==="
