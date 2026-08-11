#!/usr/bin/env bash
# Did the killed rank-1 retry (repair #5, the iris hipIpcMemHandle_t alias) get
# far enough to say anything before it was stopped?
set -uo pipefail
L=/home/subvadla/dhk/.node/rank1d.log
D=/home/subvadla/dhk/.node/compbench/rank1

echo "===== rank1d.log ====="
if [ -f "$L" ]; then
  wc -l "$L"
  echo "--- alias applied? ---"
  grep -n 'aliased iris.hip' "$L" | head -3
  echo "--- how far did it get ---"
  grep -nE 'eval.py (test|benchmark)|exit=|popcorn|test-count|benchmark-count|test\.[0-9]+\.status|check:' "$L" | head -25
else
  echo "absent"
fi

echo
echo "===== per-pass popcorn output ====="
for f in "$D"/warm.popcorn.txt "$D"/test.popcorn.txt "$D"/bench.popcorn.txt; do
  [ -f "$f" ] && { echo "--- $f ---"; head -20 "$f"; }
done

echo
echo "===== last error signature per pass ====="
for f in "$D"/warm.stderr.txt "$D"/test.stderr.txt "$D"/bench.stderr.txt; do
  [ -f "$f" ] && { echo "--- $(basename "$f") ---"; grep -E 'Error|error:|Timeout|Traceback' "$f" | tail -4; }
done

echo
echo "===== did rank-1's helpers ever write their heap bases? ====="
ls -la "$D"/heap_bases_*.pkl 2>/dev/null || echo "no heap_bases_*.pkl - the helper never completed"
