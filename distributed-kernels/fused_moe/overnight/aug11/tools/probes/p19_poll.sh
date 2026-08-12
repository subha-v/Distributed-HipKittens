#!/usr/bin/env bash
echo "=== driver out ==="
cat ~/overnight-scratch/screen_a11base.out
echo
echo "=== csv ==="
cat ~/overnight-scratch/screen_a11base.csv 2>/dev/null || echo "(no csv yet)"
echo
echo "=== alive? ==="
pgrep -af 'screen.sh|run_campaign|docker run' | head -5 || echo "(driver finished)"
exit 0
