#!/usr/bin/env bash
# Poll the detached exp_23 campaign.
#   nsh.ps1 -Script ...\exp_23_waterfall\campaign_status.sh -ArgLine "60"
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_23_waterfall
LINES=${1:-60}

if pgrep -f 'exp_23_waterfall/campaign.sh' >/dev/null 2>&1; then
  echo "STATE: campaign running"
else
  echo "STATE: campaign not running"
fi
bash "$ON/tools/gpu_lease.sh" status

echo
echo "-- draws so far --"
ls -la "$EXP"/draw_*.json 2>/dev/null || echo "  (none yet)"
ls -la "$EXP"/waterfall.json "$EXP"/stats.json 2>/dev/null

echo
echo "-- campaign.log (last $LINES lines) --"
tail -n "$LINES" "$EXP/campaign.log" 2>/dev/null || echo "  (no log)"
