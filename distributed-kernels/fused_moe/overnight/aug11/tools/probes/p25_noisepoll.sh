#!/usr/bin/env bash
echo "=== a11noise driver out ==="
cat ~/overnight-scratch/screen_a11noise.out
echo
echo "=== alive? ==="
pgrep -af 'screen.sh a11noise|run_campaign' | head -3 || echo "(finished)"
exit 0
