#!/usr/bin/env bash
awk "/<<'PY'/{f=1;next} /^PY\$/{f=0} f" ~/.overnight-scripts/screen.sh > /tmp/screenparse.py
L=$HOME/overnight-scratch/dec01_m2_C8g2fr16.log
echo "=== gate/soak/control lines in the failing log ==="
grep -h 'MOK GATE\|MPS SOAK\|control_fails\|MARK. eager' $L | head
echo
echo "=== parser verdict ==="
python3 /tmp/screenparse.py 0 "C=8,g=2,mode=2,flush_rows=16" "w1t1p1" \
  /nonexistent/summary.json "$L" /tmp/selftest3.csv 3 "a|b" "a|b" /tmp/out
echo
echo "=== CSV parseability check (all rows 35 fields?) ==="
python3 - <<'EOF'
import csv
for path in ("/tmp/selftest.csv","/tmp/selftest2.csv","/tmp/selftest3.csv"):
    try:
        rows=list(csv.reader(open(path)))
        print(path, "rows=",len(rows), "widths=", sorted({len(r) for r in rows}))
    except Exception as e:
        print(path,"ERR",e)
EOF
exit 0
