#!/usr/bin/env bash
awk "/<<'PY'/{f=1;next} /^PY\$/{f=0} f" ~/.overnight-scripts/screen.sh > /tmp/screenparse.py
L=$HOME/overnight-scratch/ts2_C64g1mode2flush_rows16timestamps1.log
echo "=== raw stamp lines in that log ==="
grep -h 'MPS TS SPLIT\|MPS TS DELTA\|MPS SPIN' $L
echo
echo "=== parser output (summary.json deliberately absent) ==="
python3 /tmp/screenparse.py 0 "C=64,g=1,mode=2,flush_rows=16,timestamps=1" "w1t1p1" \
  /nonexistent/summary.json "$L" /tmp/selftest2.csv 0 "a|b" "a|b" /tmp/out
echo
echo "=== a FAILING log, if one exists ==="
for f in $HOME/overnight-scratch/*.log; do
  if grep -q 'pass=False' "$f" && grep -q 'MOK GATE' "$f"; then echo "found: $f"; break; fi
done
exit 0
