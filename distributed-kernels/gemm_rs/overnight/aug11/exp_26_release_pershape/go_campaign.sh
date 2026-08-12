#!/usr/bin/env bash
# Launch campaign.sh detached, under setsid, so an ssh drop cannot orphan an
# 8-GPU job. The lease is acquired inside campaign.sh, not here.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign.log
mkdir -p "$D/logs"

if pgrep -f 'exp_26_release_pershape/campaign.sh' >/dev/null 2>&1; then
  echo "REFUSED: campaign already running"; exit 1
fi

echo "-- lease before --"
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -4

: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 14400 \
  bash "$D/campaign.sh" "${1:-7}" "${2:-50}" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 20
tail -15 "$LOG"
