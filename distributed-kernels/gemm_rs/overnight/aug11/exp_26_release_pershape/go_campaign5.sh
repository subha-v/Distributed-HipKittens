#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign5.log
mkdir -p "$D/logs"

echo "-- stop campaign4, which is spinning on the broken lease tool --"
pkill -TERM -f 'exp_26_release_pershape/campaign4.sh'
sleep 3
pgrep -af 'exp_26_release_pershape/campaign4.sh' || echo "  campaign4 gone"

echo "-- did campaign4 leave a lease behind? --"
if [ -d "$ON/aug11/.gpu_lease" ]; then
  echo "  held by: $(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null)"
  if [ "$(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null)" = "exp_26" ]; then
    echo "  ours and orphaned -- removing"
    rm -rf "$ON/aug11/.gpu_lease"
  fi
else
  echo "  lease FREE"
fi

pgrep -f 'exp_26_release_pershape/campaign5.sh' >/dev/null 2>&1 && {
  echo "REFUSED: campaign5 already running"; exit 1; }
: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 16200 \
  bash "$D/campaign5.sh" "${1:-9000}" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 25
tail -20 "$LOG"
