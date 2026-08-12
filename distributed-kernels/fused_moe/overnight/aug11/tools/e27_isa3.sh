#!/usr/bin/env bash
# exp_27 G4: the standing budget is "every remaining spill byte must be OUTSIDE
# both MFMA K-loops". A1 reports one more VGPR spill than A0 (15 vs 14) at an
# identical ScratchSize (128 B) and an identical static scratch-op count (19), so
# what has to be shown is PLACEMENT, not count.
#
# Method: locate the two MFMA spans by instruction address, then check whether any
# scratch_load/scratch_store address falls inside either span. CPU-only.
set -uo pipefail
O=$HOME/overnight-scratch/e27/out

for a in A0g A1g; do
  echo "################ $a ################"
  [ -f "$O/$a.isa" ] || { echo "(missing $O/$a.isa)"; continue; }

  # Instruction address is the hex after the trailing '// ' comment objdump emits.
  awk '
    function addr(s) { if (match(s, /\/\/ 0*[0-9A-F]+:/)) { t=substr(s, RSTART+3, RLENGTH-4); return strtonum("0x" t) } return -1 }
    /^; .*\.(cpp|hip|cuh|hpp|h):/ { split($0,p,":"); cur=p[length(p)]+0; f=$2; sub(/.*\//,"",f); curf=f }
    /^[[:space:]]+v_mfma/    { a=addr($0); if (a>=0) { print "MFMA", a } }
    /^[[:space:]]+scratch_/  { a=addr($0); if (a>=0) { print "SCR", a, curf ":" cur, $1 } }
  ' "$O/$a.isa" > /tmp/e27_$a.marks

  echo "--- MFMA spans (clusters separated by > 4096 B of non-MFMA code) ---"
  awk '$1=="MFMA"{print $2}' /tmp/e27_$a.marks | sort -n | awk '
    NR==1 { lo=$1; hi=$1; n=1; next }
    { if ($1-hi > 4096) { printf "  span %d: 0x%x .. 0x%x   (%d mfma, %d B)\n", ++s, lo, hi, n, hi-lo; lo=$1; n=0 }
      hi=$1; n++ }
    END { printf "  span %d: 0x%x .. 0x%x   (%d mfma, %d B)\n", ++s, lo, hi, n, hi-lo }
  '

  echo "--- scratch ops: address, source line, opcode ---"
  awk '$1=="SCR"{printf "  0x%-8x %-46s %s\n", $2, $3, $4}' /tmp/e27_$a.marks

  echo "--- G4 VERDICT: any scratch op inside an MFMA span? ---"
  awk '
    $1=="MFMA" { m[++nm]=$2 }
    $1=="SCR"  { s[++ns]=$2; sl[ns]=$3 }
    END {
      # rebuild spans exactly as above
      x=0; for(i=1;i<=nm;i++) v[i]=m[i]
      n=asort(v)
      lo=v[1]; hi=v[1]; ns_=0
      for(i=2;i<=n;i++){ if (v[i]-hi>4096){ slo[++ns_]=lo; shi[ns_]=hi; lo=v[i] } hi=v[i] }
      slo[++ns_]=lo; shi[ns_]=hi
      bad=0
      for(j=1;j<=ns;j++) for(k=1;k<=ns_;k++)
        if (s[j]>=slo[k] && s[j]<=shi[k]) { printf "  ** INSIDE span %d: 0x%x (%s)\n", k, s[j], sl[j]; bad++ }
      if (bad==0) print "  G4 PASS: zero scratch ops inside either MFMA span"
      else print "  G4 FAIL: " bad " scratch op(s) inside an MFMA span"
    }
  ' /tmp/e27_$a.marks
  echo
done
echo "===DONE==="
