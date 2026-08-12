#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign3.log
mkdir -p "$D/logs"
pgrep -f 'exp_26_release_pershape/campaign3.sh' >/dev/null 2>&1 && {
  echo "REFUSED: campaign3 already running"; exit 1; }
tr -d '\r' < "$ON/tools/gpu_lease.sh" > /tmp/gpu_lease_exp26.sh
echo "-- lease --"; bash /tmp/gpu_lease_exp26.sh status 2>&1 | head -4
echo "-- arms present --"
ls -la $ON/harness/build/gemm_rs_mi300x_{ps0,ps2,rg2c,ps0b}.so 2>&1 | awk '{print $5, $NF}'
: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 14400 \
  bash "$D/campaign3.sh" "${1:-40}" "${2:-100}" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 25
tail -20 "$LOG"
