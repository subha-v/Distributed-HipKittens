#!/usr/bin/env bash
# exp_34 step 2: the RESOURCE GATE. Compare arm B (exp_34 candidate) against arm
# A (pristine ca5b683f) built from the same tree with the same flags. NO GPU WORK.
#
# Pass/fail:
#   (1) the kernel-resource-usage remark for k0pf6gm_mps_mega must be IDENTICAL
#       between A and B -- mode 14 is present but unused, so the existing modes'
#       tuple must not move at all;
#   (2) MFMA census and flat_atomic_pk_add_bf16 census must be identical;
#   (3) zero scratch ops inside either MFMA span.
set -uo pipefail
SC=$HOME/e34
O=$SC/out

echo "############ (1) RESOURCE REMARKS ############"
for v in A B; do
  echo "======== arm $v : k0pf6gm_mps_mega remark (verbatim) ========"
  awk '/k0pf6gm_mps_mega/{f=1} f&&/remark:/{print} /Occupancy \[waves\/SIMD\]/{if(f){f=0}}' \
    "$O/$v.log" | sed 's/^.*remark: */  /'
done

echo
echo "======== normalized tuple diff (A vs B) ========"
for v in A B; do
  awk '/remark:/{ sub(/^.*remark: */,""); print }' "$O/$v.log" \
    | grep -vE '^Function Name' > "$O/$v.tuple"
done
if diff -u "$O/A.tuple" "$O/B.tuple" > "$O/tuple.diff"; then
  echo "GATE1: PASS -- resource remarks byte-identical"
else
  echo "GATE1: DIFFERS"
  cat "$O/tuple.diff"
fi
echo "--- the full remark set, arm B ---"
cat "$O/B.tuple"

echo
echo "############ (2) ISA CENSUS ############"
docker exec subha_k1 bash -lc '
set -uo pipefail
SC=/home/subvadla/e34; O=$SC/out
for v in A B; do
  llvm-objdump -d --mcpu=gfx950 $O/$v.hsaco > $O/$v.isa 2>/dev/null
done
ls -la $O/*.isa
'
for v in A B; do
  printf '%s: ' "$v"
  for m in v_mfma flat_atomic_pk_add_bf16 scratch_load scratch_store \
           buffer_wbl2 buffer_inv s_memrealtime global_atomic; do
    printf '%s=%s ' "$m" "$(grep -cE "^[[:space:]]+${m}" "$O/$v.isa")"
  done
  printf 'lines=%s\n' "$(wc -l < "$O/$v.isa")"
done

echo
echo "############ (3) SCRATCH OPS INSIDE THE MFMA SPANS ############"
# A "MFMA span" = a maximal run of instructions bracketed by the first and last
# v_mfma of a contiguous MFMA region (the two GEMM K-loops). Report any
# scratch_load/scratch_store whose line number falls between the first and last
# v_mfma of each span, where a span break is >2000 lines with no v_mfma.
for v in A B; do
  echo "---- arm $v ----"
  awk '
    /^[[:space:]]+v_mfma/ { m[++n]=NR }
    /^[[:space:]]+scratch_(load|store)/ { s[++k]=NR }
    END {
      # split MFMA line numbers into spans
      spans=0; start=m[1]; prev=m[1]
      for (i=2;i<=n;i++) {
        if (m[i]-prev > 2000) { spans++; lo[spans]=start; hi[spans]=prev; start=m[i] }
        prev=m[i]
      }
      spans++; lo[spans]=start; hi[spans]=prev
      for (j=1;j<=spans;j++) {
        c=0
        for (i=1;i<=k;i++) if (s[i]>=lo[j] && s[i]<=hi[j]) c++
        printf "  span %d: lines %d..%d  mfma_in_span=?  scratch_ops_inside=%d\n", j, lo[j], hi[j], c
      }
      printf "  total v_mfma=%d  total scratch ops=%d  spans=%d\n", n, k, spans
    }' "$O/$v.isa"
done

echo
echo "############ (4) mode-14 code is present ############"
grep -c 'K0P6_MPS_ERR_M7DONE' "$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
echo "--- .text size / fingerprint ---"
docker exec subha_k1 bash -lc '
SC=/home/subvadla/e34; O=$SC/out
for v in A B; do
  llvm-objcopy --dump-section=.text=$O/$v.text $O/$v.hsaco /dev/null 2>/dev/null
  printf "%s .text " $v; stat -c %s $O/$v.text 2>/dev/null; sha256sum $O/$v.text 2>/dev/null
done'
echo "===DONE==="
