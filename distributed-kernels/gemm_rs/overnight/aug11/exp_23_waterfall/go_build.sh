#!/usr/bin/env bash
# Host-side launcher for the exp_23 rung build. Run from Windows with:
#   powershell -ExecutionPolicy Bypass -File \
#     distributed-kernels\gemm_rs\overnight\tools\nsh.ps1 \
#     -Script distributed-kernels\gemm_rs\overnight\aug11\exp_23_waterfall\go_build.sh
#
# Detached with setsid + timeout so an ssh drop cannot kill a 4-module hipcc
# run half way through and leave three rungs built and one stale -- which is
# precisely the masquerade the fingerprint gate exists to catch, and it is
# cheaper not to create it. Poll with build_status.sh.
#
# CPU only: hipcc needs no GPU, so this does not contend with whoever holds
# the node's 8-GPU lease.
set -uo pipefail

EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
LOG=$EXP/build.log

mkdir -p "$EXP/build"
: > "$LOG"
echo "exp_23 build started $(date -Is)" >> "$LOG"

setsid timeout 7200 bash -c \
  "docker exec dhk-gemmrs bash $EXP/build_rungs.sh; \
   echo '===== BUILD EXIT '\$?' ====='; \
   docker exec -w $EXP dhk-gemmrs python3 $EXP/fingerprint.py; \
   echo '===== FINGERPRINT EXIT '\$?' ====='" \
  >> "$LOG" 2>&1 &

echo "launched pid $! -> $LOG"
sleep 3
tail -5 "$LOG"
