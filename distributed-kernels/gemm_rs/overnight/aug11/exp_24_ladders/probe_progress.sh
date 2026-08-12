#!/usr/bin/env bash
# Read-only progress probe for the exp_24 re-measure. No GPU work.
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
echo "=== runner tail ==="
tail -12 "$D/logs/remeasure_run.log" 2>/dev/null
echo
echo "=== per-shape logs (shape: last interesting line) ==="
for f in "$D"/logs/ladder_s*.log; do
  [ -f "$f" ] || continue
  printf '%-22s %s\n' "$(basename "$f")" \
    "$(grep -E 'arm-order mode|rep [0-9]|geomean|best=|Traceback|Error' "$f" 2>/dev/null | tail -1 | cut -c1-110)"
done
echo
echo "=== completed rank files ==="
ls "$D"/raw/ladder/lad_s*.rank*.json 2>/dev/null | wc -l
ls -la "$D"/raw/ladder/ 2>/dev/null | grep -c 'rank0.json'
echo
echo "=== arm order actually used (first shape that logged it) ==="
grep -h 'arm-order mode' "$D"/logs/ladder_s*.log 2>/dev/null | head -2
echo
echo "=== is it alive? ==="
pgrep -af 'ladder_mp.py' | head -3
