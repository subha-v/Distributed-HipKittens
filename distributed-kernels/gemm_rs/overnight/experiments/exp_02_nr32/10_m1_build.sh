#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/exp_02_nr32/logs
mkdir -p "$LOG"
echo "### M1 build START $(date -u +%FT%TZ)"
setsid -w timeout 1800 docker exec dhk-gemmrs bash $ON/harness/build.sh 2>&1 | tee $LOG/m1_build.log
rc=${PIPESTATUS[0]}
echo "### M1 build END rc=$rc $(date -u +%FT%TZ)"
exit $rc