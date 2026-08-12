#!/usr/bin/env bash
# exp_34 step 8: WHERE do the scratch accesses live? The gate that matters is not
# the count, it is whether any scratch op sits inside an MFMA span or in front of
# the remote-atomic epilogue -- that is the exp_26 regression signature (a
# scratch_load 17 instructions ahead of 96 of the 282 flat_atomic_pk_add_bf16).
# NO GPU WORK.
set -uo pipefail
SC=$HOME/e34; O=$SC/out

for v in A B N8; do
  [ -s "$O/$v.s" ] || { echo "$v: no .s"; continue; }
  echo "################ arm $v ################"
  awk '
    /^[[:space:]]*[a-zA-Z_.][a-zA-Z0-9_.$]*:/ { next }              # labels
    /^[[:space:]]+v_mfma/                     { ++ni; k[ni]="M"; ln[ni]=NR; ++nm; mline[nm]=NR }
    /^[[:space:]]+flat_atomic_pk_add_bf16/    { ++ni; k[ni]="A"; ln[ni]=NR; ++na; aline[na]=NR }
    /^[[:space:]]+scratch_(load|store)/       { ++ni; k[ni]="S"; ln[ni]=NR; ++ns; sline[ns]=NR }
    END {
      if (nm==0) { print "  no mfma"; exit }
      # --- MFMA spans (gap > 2000 asm lines starts a new span) ---
      sp=0; start=mline[1]; prev=mline[1]; cnt=1
      for (i=2;i<=nm;i++) {
        if (mline[i]-prev > 2000) { sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt; start=mline[i]; cnt=0 }
        prev=mline[i]; cnt++
      }
      sp++; lo[sp]=start; hi[sp]=prev; mc[sp]=cnt
      for (j=1;j<=sp;j++) {
        c=0; for (i=1;i<=ns;i++) if (sline[i]>=lo[j] && sline[i]<=hi[j]) c++
        printf "  MFMA span %d: asm lines %6d..%6d  v_mfma=%3d  >>> SCRATCH OPS INSIDE = %d <<<\n", j, lo[j], hi[j], mc[j], c
      }
      # --- atomics region ---
      printf "  atomics: n=%d  first=%d last=%d\n", na, aline[1], aline[na]
      c=0; for (i=1;i<=ns;i++) if (sline[i]>=aline[1] && sline[i]<=aline[na]) c++
      printf "  scratch ops between the FIRST and LAST remote atomic = %d\n", c
      # --- how many atomics have a scratch op within 40 asm lines ahead ---
      h=0
      for (i=1;i<=na;i++) {
        for (j=1;j<=ns;j++) if (sline[j] < aline[i] && aline[i]-sline[j] <= 40) { h++; break }
      }
      printf "  atomics with a scratch op within 40 asm lines AHEAD = %d of %d  (exp_26 signature)\n", h, na
      # --- distribution: before first mfma / inside / between spans / after last ---
      b=0; aft=0
      for (i=1;i<=ns;i++) {
        if (sline[i] < lo[1]) b++
        else if (sline[i] > hi[sp]) aft++
      }
      printf "  scratch ops: before first MFMA=%d   after last MFMA=%d   total=%d\n", b, aft, ns
    }' "$O/$v.s"
done
echo "===DONE==="
