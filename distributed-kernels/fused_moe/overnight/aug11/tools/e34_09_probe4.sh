#!/usr/bin/env bash
# exp_34 step 9: which piece re-triggers exp_26's scratch migration into the
# remote-atomic epilogue (96 of 282 atomics with a scratch op <=40 lines ahead)?
#   W1 = N8 with the Ready=true M8 instantiation reverted to mode 12's
#        (i.e. NO fifth inlined combine body). Probe only -- mode 14 would hang.
#   W2 = pristine kernel + the new adapter only (the plumbing floor).
#   W3 = N8 with the coarse publish arm removed.
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out

python3 - "$SC" <<'PY'
import sys, pathlib
sc = pathlib.Path(sys.argv[1])
n8 = (sc / "tu/N8/k0pf6gm_device_tile_mps.hip").read_text()
READY_INST = "false, true, true>"
assert n8.count(READY_INST) == 1
PUB = """      if (coarse75) {
        if (bid == 0) {
          unsigned int* self_slot =
              row_ready + (size_t)cur * T_loc_max + (size_t)T_ext;
          for (int R = 0; R < world; ++R) {
            if (R == cur) {
              hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);
            } else {
              hk_moe::publish_epoch<hk_moe::scope::system>(
                  hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
            }
          }
        }
      } else {
"""
assert n8.count(PUB) == 1
out = {
    "W1": n8.replace(READY_INST, "false, true>"),
    "W3": n8.replace(PUB, "      {\n"),
}
for name, text in out.items():
    d = sc / "tu" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "k0pf6gm_device_tile_mps.hip").write_text(text)
    print(name, "written", len(text))
# W2: pristine kernel + new adapter, in its own include dir
d = sc / "tu/W2"
d.mkdir(parents=True, exist_ok=True)
(d / "k0pf6gm_device_tile_mps.hip").write_text(
    (sc / "base/k0pf6gm_device_tile_mps.hip.orig").read_text())
(d / "moe_mps_adapter.cuh").write_text(
    (sc / "DHK/distributed-kernels/fused_moe/moe_mps_adapter.cuh").read_text())
print("W2 written")
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
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in W1 W2 W3; do
  # W2 must see its OWN adapter first on the include path
  hipcc "${COMMON[@]}" -I$SC/tu/$v -I$DHK/include -I$FM \
    --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  echo "$v exit=$? errors=$(grep -cE "error:" $O/$v.log)"
  hipcc "${COMMON[@]}" -I$SC/tu/$v -I$DHK/include -I$FM \
    --cuda-device-only -S $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
done
echo asm_done'

echo
printf '%-4s %-8s %-11s %-11s %-9s %-8s %-8s\n' arm scratch sgpr_spill vgpr_spill scr_load mfma pk_add
for v in A N8 W1 W2 W3; do
  f=$O/$v.log; s=$O/$v.s
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  printf '%-4s %-8s %-11s %-11s %-9s %-8s %-8s\n' "$v" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" \
    "$(c scratch_load)" "$(c v_mfma)" "$(c flat_atomic_pk_add_bf16)"
done

echo
echo "############ exp_26 SIGNATURE ############"
for v in A N8 W1 W2 W3; do
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
      printf "scratch_in_mfma_spans=%d  scratch_between_atomics=%d  atomics_with_scratch_ahead=%d/%d\n", inm, btw, h, na
    }' "$O/$v.s"
done
echo "===DONE==="
