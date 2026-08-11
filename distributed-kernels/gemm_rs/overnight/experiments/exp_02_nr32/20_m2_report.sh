#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/exp_02_nr32/logs
mkdir -p "$LOG"
echo "### M2 report START $(date -u +%FT%TZ)"
setsid -w timeout 900 docker exec dhk-gemmrs bash $ON/tools/m2_report.sh 2>&1 | tee $LOG/m2_report.log
rc=${PIPESTATUS[0]}
echo "### M2 report END rc=$rc $(date -u +%FT%TZ)"
exit $rc