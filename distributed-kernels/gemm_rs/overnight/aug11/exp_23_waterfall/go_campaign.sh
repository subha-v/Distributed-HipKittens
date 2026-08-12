#!/usr/bin/env bash
# Host-side launcher for the exp_23 GPU campaign. Detached with setsid so an
# ssh drop cannot orphan a half-finished set of draws while still holding the
# node's GPU lease.
#   nsh.ps1 -Script ...\exp_23_waterfall\go_campaign.sh
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
LOG=$EXP/campaign.log

: > "$LOG"
setsid timeout 12000 bash "$EXP/campaign.sh" >> "$LOG" 2>&1 &
echo "launched pid $! -> $LOG"
sleep 20
tail -20 "$LOG"
