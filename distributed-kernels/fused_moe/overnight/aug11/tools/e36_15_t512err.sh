#!/usr/bin/env bash
# exp_36: why did T=512 abort before producing rank JSONs?
set -u
L="$(ls -t $HOME/e36/scratch/e36T512_1_*.log 2>/dev/null | head -1)"
echo "log: $L"
grep -n 'exp_36 shape' "$L" | head -2
echo "=== first error-ish lines ==="
grep -n -i 'error\|Error\|Traceback\|raise \|ValueError\|RuntimeError\|assert' "$L" | head -30
echo "=== traceback context ==="
awk '/Traceback/{f=1} f{print NR": "$0} f&&/^\s*$/{c++; if(c>2) exit}' "$L" | head -60
echo "=== tail ==="
tail -25 "$L"
exit 0
