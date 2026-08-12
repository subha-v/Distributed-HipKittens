#!/usr/bin/env bash
# t08: poll the current screen batch. Pass TAG in E32TAG (default e32regate).
set -uo pipefail
TAG="${E32TAG:-e32regate}"
LOG="$HOME/overnight-scratch/${TAG}.driver.log"
echo "== alive? =="
pgrep -af 'screen.sh|torchrun|run_campaign' | head -10 || echo "(no driver/job)"
echo "== driver log =="
tail -30 "$LOG" 2>/dev/null || echo "(no driver log $LOG)"
echo "== per-run logs =="
ls -1t "$HOME/overnight-scratch/${TAG}_"*.log 2>/dev/null | head -6
for f in $(ls -1t "$HOME/overnight-scratch/${TAG}_"*.log 2>/dev/null | head -2); do
  echo "---- $f ($(wc -l < "$f") lines) ----"
  grep -E '\[POISON|\[MOK GATE\]|\[MPS SOAK\]|\[MARK\] control_fails|\[MARK\] eager|\[MPS TS SPLIT\]|\[MPS TS DELTA\]|arm_p50|Traceback|Error|error:|blocked' "$f" | tail -40
  echo "   (tail)"; tail -4 "$f"
done
echo "== csv =="
cat "$HOME/overnight-scratch/screen_${TAG}.csv" 2>/dev/null
exit 0
