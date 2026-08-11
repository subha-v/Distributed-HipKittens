#!/usr/bin/env bash
# M2 prerequisites: tools/m2_report.sh reads $ON/build/m1a.log and $ON/build/isa/*.s,
# which are produced by tools/m1_build.sh and tools/m2_isa.sh. Neither had been run
# in this synced tree, so build/ did not exist. Compile-only; no GPU involvement.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/exp_02_nr32/logs
mkdir -p "$LOG"

echo "### m1_build.sh START $(date -u +%FT%TZ)"
setsid -w timeout 2400 docker exec dhk-gemmrs bash $ON/tools/m1_build.sh 2>&1 | tee $LOG/m2_prereq_m1_build.log
rc1=${PIPESTATUS[0]}
echo "### m1_build.sh END rc=$rc1 $(date -u +%FT%TZ)"
[ "$rc1" != "0" ] && exit $rc1

echo
echo "### m2_isa.sh START $(date -u +%FT%TZ)"
setsid -w timeout 2400 docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh 2>&1 | tee $LOG/m2_isa.log
rc2=${PIPESTATUS[0]}
echo "### m2_isa.sh END rc=$rc2 $(date -u +%FT%TZ)"
exit $rc2