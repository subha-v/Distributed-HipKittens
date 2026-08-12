#!/usr/bin/env bash
# exp_28 probe 3 (read-only): exp_65 (kChunks 8->16) -- the measured traffic-vs-latency
# discriminator, plus any later M6 attribution work.
set -uo pipefail
AM="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"

echo "===E65_DIR==="
ls -d "$AM"/experiments/*exp_65* "$AM"/experiments/*exp_6[6-9]* "$AM"/experiments/*exp_7*  2>/dev/null

echo "===E65_RESULT==="
cat "$AM"/experiments/exp_65*/result.md 2>/dev/null

echo "===SUMMARY_E65_PARA==="
awk '/^## exp_65/,/^## exp_67/' "$AM/experiments/summary.md" | head -80

echo "===LATEST_SUMMARY_TAIL==="
tail -60 "$AM/experiments/summary.md"
echo "===DONE==="
