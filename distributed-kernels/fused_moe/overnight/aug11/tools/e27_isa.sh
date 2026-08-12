#!/usr/bin/env bash
# exp_27 G5: find the M6 ascale gather in the ISA and prove the line-traffic cut.
# The whole-kernel mnemonic counts came out nearly identical, which is suspicious,
# so first establish what the loads are actually SPELLED as, then locate the
# gather by its LDS-write signature and print it.
set -uo pipefail
SC=$HOME/overnight-scratch/e27
O=$SC/out

echo "===MNEMONIC HISTOGRAM (top 40, A0)==="
awk '{ if (match($0, /^\s+[a-z][a-z0-9_]+/)) { m=$1; c[m]++ } } END { for (k in c) print c[k], k }' \
  "$O/A0.isa" | sort -rn | head -40

echo
echo "===LOAD-FAMILY COUNTS, per arm==="
for v in R A0 A1; do
  printf '%-3s ' "$v"
  for m in buffer_load_dword buffer_load_dwordx2 buffer_load_dwordx4 \
           global_load_dword global_load_dwordx2 global_load_dwordx4 \
           flat_load_dword flat_load_dwordx4 ds_write_b32 ds_write_b128 \
           s_load_dword v_mad_u64_u32 v_mul_lo_u32; do
    printf '%s=%s ' "$m" "$(grep -cE "^[[:space:]]+${m}[[:space:]]" "$O/$v.isa")"
  done
  printf '\n'
done

echo
echo "===LOCATE THE GATHER: every basic block containing >=4 ds_write_b32 ==="
# The gather is the only site that writes FOUR consecutive ascale_lds dwords from
# one loaded quad (arm 1) or one dword per iteration (arm 0). Print a window
# around each ds_write_b32 cluster.
for v in A0 A1; do
  echo "######## $v ########"
  awk -v V="$v" '
    /ds_write_b32/ { hits[NR]=1 }
    { line[NR]=$0 }
    END {
      n=0
      for (i=1;i<=NR;i++) if (hits[i]) { c=0; for (j=i;j<i+40 && j<=NR;j++) if (hits[j]) c++
        if (c>=4 && (i-last>40)) { last=i; n++
          print "---- cluster " n " at line " i " (" c " ds_write_b32 within 40) ----"
          for (j=i-46;j<=i+22 && j<=NR;j++) if (j>0) print line[j]
          print ""
        } }
    }' "$O/$v.isa" | head -220
done
echo "===DONE==="
