#!/usr/bin/env bash
# exp_36: T=1024 with capacities pinned at the T=4096 values left the whole
# output poisoned on every arm. Find out where it went wrong.
set -u
L="$(ls -t $HOME/e36/scratch/e36T1024_1_*.log 2>/dev/null | head -1)"
echo "log: $L"
grep -n 'exp_36 shape' "$L" | head -2
echo "=== POISON / GATE / MARK ==="
grep -n 'POISON\|MOK GATE\|\[MARK\]\|survivors' "$L" | head -25
echo "=== errors ==="
grep -n -i 'Traceback\|RuntimeError\|ValueError\|AssertionError\|Error:' "$L" | head -20
echo "=== first 12 lines around first traceback ==="
n=$(grep -n 'Traceback' "$L" | head -1 | cut -d: -f1)
[ -n "${n:-}" ] && sed -n "$((n-4)),$((n+18))p" "$L"
echo "=== K0PF GATE lines ==="
grep -n '\[K0PF GATE\]' "$L" | head -15
echo "=== tail ==="
tail -20 "$L"
exit 0
