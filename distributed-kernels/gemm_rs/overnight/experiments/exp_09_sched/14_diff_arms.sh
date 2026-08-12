#!/usr/bin/env bash
# Did arm X's ISA actually differ from arm Y's? Answers "the compiler defeated
# me" with a number instead of an impression.
#   14_diff_arms.sh <armX> <armY>
set -uo pipefail
X=${1:?armX}; Y=${2:?armY}
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_09_sched
SX=$EXP/arms/$X/gemm_rs_mi300x.gfx942.s
SY=$EXP/arms/$Y/gemm_rs_mi300x.gfx942.s

echo "===== whole-file instruction-stream diff ($X vs $Y) ====="
# strip comments, labels and blank lines so line-number churn does not show up
norm() { sed 's/;.*//' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
         | grep -vE '^$|^\.|:$' ; }
norm "$SX" > /tmp/x.$$ ; norm "$SY" > /tmp/y.$$
echo "instructions: $X=$(wc -l < /tmp/x.$$)  $Y=$(wc -l < /tmp/y.$$)"
d=$(diff /tmp/x.$$ /tmp/y.$$ | grep -cE '^[<>]')
echo "differing instruction lines: $d"
if [ "$d" != "0" ]; then diff /tmp/x.$$ /tmp/y.$$ | head -40; fi
rm -f /tmp/x.$$ /tmp/y.$$

echo
echo "===== per-instantiation mainloop shape ====="
for A in "$X" "$Y"; do
  echo "-- $A --"
  grep -E 'deepest MFMA loop' "$EXP/arms/$A/kloop_256.txt" 2>/dev/null
  awk '/s_waitcnt vmcnt\(0\)/{print "   commit wait at: "$0}' \
      "$EXP/arms/$A/kloop_256.txt" 2>/dev/null | head -4
done
