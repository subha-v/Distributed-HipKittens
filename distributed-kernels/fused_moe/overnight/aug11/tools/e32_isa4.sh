#!/usr/bin/env bash
# exp_32 Job 2 step 4: cluster the flat_load_dwordx4 sites and find the M8 combine
# body (the one that is preceded by ds_bpermute_b32 pairs and followed by
# flat_store_dwordx4 zeroing). READ-ONLY.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
OUT=$HOME/overnight-scratch/e32
ISA=$SC/out2/D.isa

echo "===flat_load_dwordx4 CLUSTERS==="
grep -n 'flat_load_dwordx4' "$ISA" | cut -d: -f1 > "$OUT/fl4.lines"
awk '{a[NR]=$1} END{
  n=NR; i=1
  while (i<=n) {
    j=i
    while (j+1<=n && a[j+1]-a[j] <= 30) j++
    printf "cluster lines %6d..%6d  count=%3d  span=%5d\n", a[i], a[j], j-i+1, a[j]-a[i]
    i=j+1
  }
}' "$OUT/fl4.lines"

echo "===flat_atomic_pk_add_bf16 CLUSTERS (M7 epilogue, mode 12)==="
grep -n 'flat_atomic_pk_add_bf16' "$ISA" | cut -d: -f1 \
 | awk '{a[NR]=$1} END{n=NR;i=1; while(i<=n){j=i; while(j+1<=n && a[j+1]-a[j]<=40) j++;
   printf "cluster lines %6d..%6d count=%3d\n", a[i], a[j], j-i+1; i=j+1}}'

echo "===ds_bpermute_b32 CLUSTERS==="
grep -n 'ds_bpermute_b32' "$ISA" | cut -d: -f1 \
 | awk '{a[NR]=$1} END{n=NR;i=1; while(i<=n){j=i; while(j+1<=n && a[j+1]-a[j]<=30) j++;
   printf "cluster lines %6d..%6d count=%3d\n", a[i], a[j], j-i+1; i=j+1}}'

echo "===flat_store_dwordx4 CLUSTERS==="
grep -n 'flat_store_dwordx4' "$ISA" | cut -d: -f1 \
 | awk '{a[NR]=$1} END{n=NR;i=1; while(i<=n){j=i; while(j+1<=n && a[j+1]-a[j]<=60) j++;
   printf "cluster lines %6d..%6d count=%3d\n", a[i], a[j], j-i+1; i=j+1}}'

echo "===v_mfma CLUSTERS (M6 then M7 K-loops, for orientation)==="
grep -n 'v_mfma' "$ISA" | cut -d: -f1 \
 | awk '{a[NR]=$1} END{n=NR;i=1; while(i<=n){j=i; while(j+1<=n && a[j+1]-a[j]<=200) j++;
   printf "cluster lines %6d..%6d count=%3d\n", a[i], a[j], j-i+1; i=j+1}}'

echo "===LAST global_atomic_or SITES (pperr; M8/M9 tail orientation)==="
grep -n 'global_atomic_or' "$ISA" | tail -12
exit 0
