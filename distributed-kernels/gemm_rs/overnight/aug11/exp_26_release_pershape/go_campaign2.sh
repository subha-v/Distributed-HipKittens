#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
LOG=$D/logs/campaign2.log
mkdir -p "$D/logs"
pgrep -f 'exp_26_release_pershape/campaign2.sh' >/dev/null 2>&1 && {
  echo "REFUSED: campaign2 already running"; exit 1; }
echo "-- source default (must be 0; arms carry explicit -D) --"
grep -A1 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
echo "-- lease (via a CR-stripped copy; the shared tool is CRLF-broken) --"
tr -d '\r' < "$ON/tools/gpu_lease.sh" > /tmp/gpu_lease_exp26.sh
bash /tmp/gpu_lease_exp26.sh status 2>&1 | head -4
: > "$LOG"
setsid nohup timeout --signal=TERM --kill-after=600 14400 \
  bash "$D/campaign2.sh" "${1:-25}" "${2:-100}" >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 25
tail -25 "$LOG"
