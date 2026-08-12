#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign6.log
mkdir -p "$D/logs"
if [ -d "$ON/aug11/.gpu_lease" ]; then
  echo "lease held by: $(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null)"
  if [ "$(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null)" = "exp_26" ]; then
    echo "  ours and orphaned -- removing"; rm -rf "$ON/aug11/.gpu_lease"
  fi
else
  echo "lease FREE"
fi
pgrep -f 'exp_26_release_pershape/campaign6.sh' >/dev/null 2>&1 && {
  echo "REFUSED: campaign6 already running"; exit 1; }
: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 18000 \
  bash "$D/campaign6.sh" "${1:-9000}" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 30
tail -30 "$LOG"
