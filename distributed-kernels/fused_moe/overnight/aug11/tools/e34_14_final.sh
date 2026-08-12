#!/usr/bin/env bash
# exp_34 step 14: FINAL gate on the delivered files. Rebuild A (pristine
# ca5b683f) and B (candidate) from one tree with identical flags, capture the
# resource remark verbatim, and run the full ISA census. NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
FM=$SC/DHK/distributed-kernels/fused_moe

echo "=== delivered file hashes (node scratch clone) ==="
sha256sum "$FM/k0pf6gm_device_tile_mps.hip" "$FM/moe_mps_adapter.cuh"
cd "$SC/DHK"; git diff --stat ca5b683f -- distributed-kernels/fused_moe/
git diff ca5b683f -- distributed-kernels/fused_moe/ > "$O/e34_final.diff"
echo "diff lines: $(wc -l < "$O/e34_final.diff")"
echo "SRC_REV: $(grep -m1 'define K0P6_MPS_SRC_REV' "$FM/k0pf6gm_device_tile_mps.hip")"

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
echo "### FINAL_B (candidate, as delivered)"
hipcc "${COMMON[@]}" -I$DHK/include -I$FM --genco \
  $FM/k0pf6gm_device_tile_mps.hip -o $O/FB.hsaco > $O/FB.log 2>&1
echo "exit=$? errors=$(grep -cE "error:" $O/FB.log)"
hipcc "${COMMON[@]}" -I$DHK/include -I$FM --cuda-device-only -S \
  $FM/k0pf6gm_device_tile_mps.hip -o $O/FB.s > /dev/null 2>&1
echo "### FINAL_A (pristine ca5b683f, own include dir)"
mkdir -p $SC/atu
cp $SC/base/k0pf6gm_device_tile_mps.hip.orig $SC/atu/k0pf6gm_device_tile_mps.hip
cp $SC/base/moe_mps_adapter.cuh.orig         $SC/atu/moe_mps_adapter.cuh
hipcc "${COMMON[@]}" -I$SC/atu -I$DHK/include -I$FM --genco \
  $SC/atu/k0pf6gm_device_tile_mps.hip -o $O/FA.hsaco > $O/FA.log 2>&1
echo "exit=$? errors=$(grep -cE "error:" $O/FA.log)"
hipcc "${COMMON[@]}" -I$SC/atu -I$DHK/include -I$FM --cuda-device-only -S \
  $SC/atu/k0pf6gm_device_tile_mps.hip -o $O/FA.s > /dev/null 2>&1
echo done'

echo
echo "################ RESOURCE REMARK, VERBATIM ################"
for v in FA FB; do
  echo "-------- $v --------"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Dynamic Stack|Occupancy|LDS Size|Spill' "$O/$v.log" \
    | sed 's|.*remark: *||' | sed 's| \[-Rpass-analysis=kernel-resource-usage\]||'
done

echo
echo "################ GATE TABLE ################"
printf '%-24s %-14s %-14s %-8s\n' item 'A (ca5b683f)' 'B (exp_34)' verdict
row() { printf '%-24s %-14s %-14s %-8s\n' "$1" "$2" "$3" "$4"; }
ga() { grep -m1 "$1" "$O/FA.log" | sed 's|.*: *||;s| \[.*||'; }
gb() { grep -m1 "$1" "$O/FB.log" | sed 's|.*: *||;s| \[.*||'; }
ca() { grep -cE "^[[:space:]]+$1" "$O/FA.s"; }
cb() { grep -cE "^[[:space:]]+$1" "$O/FB.s"; }
cmp2() { [ "$2" = "$3" ] && echo PASS || echo "DIFF"; }
for k in TotalSGPRs 'VGPRs:' 'AGPRs:' ScratchSize 'SGPRs Spill' 'VGPRs Spill' 'LDS Size' Occupancy; do
  a=$(ga "$k"); b=$(gb "$k"); row "$k" "$a" "$b" "$(cmp2 x "$a" "$b")"
done
for m in v_mfma flat_atomic_pk_add_bf16 scratch_load scratch_store; do
  a=$(ca "$m"); b=$(cb "$m"); row "$m" "$a" "$b" "$(cmp2 x "$a" "$b")"
done

echo
echo "################ MFMA-SPAN + EPILOGUE SCRATCH ################"
for v in FA FB; do
  printf '%-4s ' "$v"
  awk '
    /^[[:space:]]+v_mfma/                  { ++nm; mline[nm]=NR }
    /^[[:space:]]+flat_atomic_pk_add_bf16/ { ++na; aline[na]=NR }
    /^[[:space:]]+scratch_(load|store)/    { ++ns; sline[ns]=NR }
    END {
      sp=0; start=mline[1]; prev=mline[1]; cnt=1
      for (i=2;i<=nm;i++) { if (mline[i]-prev>2000) { sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt; start=mline[i]; cnt=0 } prev=mline[i]; cnt++ }
      sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt
      im=0; for (j=1;j<=sp;j++) for (i=1;i<=ns;i++) if (sline[i]>=lo[j] && sline[i]<=hi[j]) im++
      h=0;  for (i=1;i<=na;i++) for (j=1;j<=ns;j++) if (sline[j]<aline[i] && aline[i]-sline[j]<=40) { h++; break }
      printf "mfma_spans=%d (v_mfma %d+%d)  SCRATCH_INSIDE_MFMA_SPANS=%d  atomics_with_scratch<=40_ahead=%d/%d  total_scratch_ops=%d\n",
             sp, mc[1], mc[2], im, h, na, ns }' "$O/$v.s"
done
echo "===DONE==="
