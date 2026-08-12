#!/usr/bin/env bash
# exp_34 step 5: the RESOURCE GATE, corrected disassembly path. NO GPU WORK.
set -uo pipefail
SC=$HOME/e34; O=$SC/out

echo "############ TUPLES ############"
printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s %-4s\n' \
  arm SGPR VGPR AGPR scratch sgpr_spill vgpr_spill LDS occ
for v in A B; do
  f=$O/$v.log
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s %-4s\n' "$v" \
    "$(g 'TotalSGPRs')" "$(g '^.*VGPRs:')" "$(g 'AGPRs:')" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" \
    "$(g 'LDS Size')" "$(g 'Occupancy')"
done
echo
echo "--- verbatim remark, arm B ---"
grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Dynamic Stack|Occupancy|LDS Size|Spill' "$O/B.log" \
  | sed 's|.*remark: *||' | sed 's| \[-Rpass-analysis=kernel-resource-usage\]||'
echo
echo "--- verbatim remark, arm A (pristine ca5b683f) ---"
grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Dynamic Stack|Occupancy|LDS Size|Spill' "$O/A.log" \
  | sed 's|.*remark: *||' | sed 's| \[-Rpass-analysis=kernel-resource-usage\]||'

echo
echo "############ DISASSEMBLE ############"
docker exec subha_k1 bash -lc '
set -uo pipefail
O=/home/subvadla/e34/out
for v in A B; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $O/$v.hsaco > $O/$v.isa 2>&1
done
wc -l $O/A.isa $O/B.isa'

echo
echo "############ CENSUS ############"
printf '%-4s %-10s %-8s %-13s %-9s %-9s %-11s %-11s\n' \
  arm isa_lines v_mfma pk_add_bf16 scr_load scr_store buffer_wbl2 buffer_inv
for v in A B; do
  printf '%-4s %-10s %-8s %-13s %-9s %-9s %-11s %-11s\n' "$v" \
    "$(wc -l < $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+v_mfma' $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+flat_atomic_pk_add_bf16' $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+scratch_load' $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+scratch_store' $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+buffer_wbl2' $O/$v.isa)" \
    "$(grep -cE '^[[:space:]]+buffer_inv' $O/$v.isa)"
done

echo
echo "############ SCRATCH OPS INSIDE THE MFMA SPANS ############"
# span = maximal run of v_mfma lines with no gap > 2000 disassembly lines
for v in A B; do
  echo "---- arm $v ----"
  awk '
    /^[[:space:]]+v_mfma/           { m[++n]=NR }
    /^[[:space:]]+scratch_(load|store)/ { s[++k]=NR }
    END {
      if (n==0) { print "  NO MFMA FOUND -- disassembly failed"; exit }
      sp=0; start=m[1]; prev=m[1]; cnt=1
      for (i=2;i<=n;i++) {
        if (m[i]-prev > 2000) { sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt; start=m[i]; cnt=0 }
        prev=m[i]; cnt++
      }
      sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt
      for (j=1;j<=sp;j++) {
        c=0; for (i=1;i<=k;i++) if (s[i]>=lo[j] && s[i]<=hi[j]) c++
        printf "  span %d: isa lines %d..%d  v_mfma=%d  SCRATCH OPS INSIDE=%d\n", j, lo[j], hi[j], mc[j], c
      }
      printf "  totals: v_mfma=%d scratch_ops=%d spans=%d\n", n, k, sp
    }' "$O/$v.isa"
done

echo
echo "############ WHERE THE SCRATCH OPS LIVE (arm B, relative to last mfma) ############"
awk '
  /^[[:space:]]+v_mfma/ { last=NR }
  /^[[:space:]]+scratch_(load|store)/ { if (NR>last) after++; else before++ }
  END { printf "  before/inside last mfma=%d   after last mfma=%d\n", before, after }' "$O/B.isa"
awk '
  /^[[:space:]]+v_mfma/ { last=NR }
  /^[[:space:]]+scratch_(load|store)/ { if (NR>last) after++; else before++ }
  END { printf "  arm A: before=%d after=%d\n", before, after }' "$O/A.isa"
echo "===DONE==="
