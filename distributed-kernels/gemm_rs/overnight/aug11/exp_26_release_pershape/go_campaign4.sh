#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign4.log
mkdir -p "$D/logs"
pgrep -f 'exp_26_release_pershape/campaign4.sh' >/dev/null 2>&1 && {
  echo "REFUSED: campaign4 already running"; exit 1; }
tr -d '\r' < "$ON/tools/gpu_lease.sh" > /tmp/gpu_lease_exp26.sh
echo "-- lease --"; bash /tmp/gpu_lease_exp26.sh status 2>&1 | head -4
: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 16200 \
  bash "$D/campaign4.sh" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 25
tail -20 "$LOG"
