#!/usr/bin/env bash
# One arm, end to end: ISA inspection first (cheap, catches spills and proves
# the schedule changed), then the full gate ladder, then archive the ladder
# logs next to the arm's ISA so nothing is overwritten by the next arm.
#   arm_full.sh <arm-name> [skip_soak]
set -uo pipefail
ARM=${1:?usage: arm_full.sh <arm-name> [skip_soak]}
SKIP=${2:-}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
D=$EXP/arms/$ARM

bash "$EXP/isa_arm.sh" "$ARM" || { echo "ARM $ARM: ISA STEP FAILED"; exit 1; }

echo
echo "#################### $ARM: GATE LADDER ####################"
bash "$EXP/run_ladder.sh" exp_09_sched $SKIP
rc=$?

mkdir -p "$D/gates"
cp -f "$EXP/logs/"m*.log "$D/gates/" 2>/dev/null || true
cp -f "$EXP/logs/m7_results.json" "$D/gates/" 2>/dev/null || true
echo
echo "#################### $ARM: SUMMARY ####################"
echo "resource tuples:"
sed -n '/BM\/BN\/BK/,/^$/p' "$EXP/logs/m2_report.log" 2>/dev/null | head -8
echo
grep -E 'shapes PASSED|GATE M4|GATE M5' "$EXP/logs/m3_correctness.log" \
     "$EXP/logs/m4_controls.log" "$EXP/logs/m5_soak.log" 2>/dev/null | tail -5
echo
sed -n '/GATE M7/,$p' "$EXP/logs/m7_bench.log" 2>/dev/null | head -22
echo "ARM $ARM rc=$rc"
exit $rc
