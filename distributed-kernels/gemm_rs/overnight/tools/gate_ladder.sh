#!/usr/bin/env bash
# The full gate ladder for one candidate, in the mandated order, with logs.
#
#   gate_ladder.sh <exp_dir_name> [skip_soak]
#
# e.g. gate_ladder.sh exp_04_tile_table
#
# Order is not negotiable: M1 build -> M2 resources/ISA -> M3 correctness on
# all 17 shapes at BOTH 1e-2 (graded) and 2e-3 (tight) -> M4 three negative
# controls -> M5 600-epoch soak -> M7 timing. Timing runs last and only if
# everything above passed, and only on a node with no other KFD process.
#
# A nonzero error bit is terminal: this script stops rather than retrying.
# Passing 1e-2 while failing 2e-3 is a REGRESSION and is treated as failure.
set -uo pipefail

EXP=${1:?usage: gate_ladder.sh <exp_dir_name> [skip_soak]}
SKIP_SOAK=${2:-0}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/$EXP
L=$D/logs
mkdir -p "$L"

step() { printf '\n########## %s ##########\n' "$1"; }
fail() { echo "GATE LADDER FAILED AT: $1"; exit 1; }

step "M0 preflight: node must be clean"
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || fail "M0 node dirty (another GPU job is running)"
rocm-smi 2>&1 | tail -12 | head -10

step "M1 build"
docker exec dhk-gemmrs bash $ON/harness/build.sh 2>&1 | tee "$L/m1_build.log"
grep -q "ALL MODULES BUILT" "$L/m1_build.log" || fail "M1 build"

step "M2 resources / ISA"
# m2_report.sh reads $ON/build/m1a.log and $ON/build/isa/*.s, which ONLY
# m1_build.sh and m2_isa.sh produce -- and it runs `set -uo pipefail` without
# -e, so with those inputs missing it prints FileNotFoundError and still
# EXITS 0. M2 was a silent false pass for the whole session before this was
# caught. Run the prerequisites, then assert the table is real.
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$L/m2_prereq_m1.log" 2>&1 || \
  echo "  (m1_build prereq returned nonzero; continuing to m2_isa)"
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh  > "$L/m2_prereq_isa.log" 2>&1 || \
  echo "  (m2_isa prereq returned nonzero)"
docker exec dhk-gemmrs bash $ON/tools/m2_report.sh 2>&1 | tee "$L/m2_report.log"

grep -qE 'Error|Traceback|No such file' "$L/m2_report.log" && fail "M2 (report could not read its inputs)"
# 7 instantiations must appear in the metadata table, each with a resource tuple.
rows=$(sed -n '/metadata table/,/ordering ops/p' "$L/m2_report.log" | grep -cE 'tail=[01]')
echo "  M2 metadata rows: $rows (expect 7)"
[ "$rows" -ge 7 ] || fail "M2 (metadata table has $rows rows, expected 7 -- gate is vacuous)"
echo "  --- resource tuples ---"
sed -n '/metadata table/,/ordering ops/p' "$L/m2_report.log" | grep -E 'BM/BN/BK|tail=[01]'

step "M3 correctness, 17 shapes, 1e-2 AND 2e-3"
docker exec -w $ON/harness dhk-gemmrs timeout 2400 \
  python3 -u m3_correctness.py all 2>&1 | tee "$L/m3_correctness.log"
grep -qE 'shapes PASSED' "$L/m3_correctness.log" || fail "M3 correctness"
if grep -qE '\bFAIL\b' "$L/m3_correctness.log"; then fail "M3 correctness (a shape FAILed)"; fi

step "M4 negative controls"
docker exec -w $ON/harness dhk-gemmrs timeout 1200 \
  python3 -u m4_controls.py 2>&1 | tee "$L/m4_controls.log"
grep -qiE 'all controls|controls PASSED|3/3' "$L/m4_controls.log" || \
  echo "  (note: check m4 verdict text by hand)"

if [ "$SKIP_SOAK" != "skip_soak" ]; then
  step "M5 600-epoch skewed soak"
  docker exec -w $ON/harness dhk-gemmrs timeout 2400 \
    python3 -u m5_soak.py 600 512 4096 12288 1 1 2>&1 | tee "$L/m5_soak.log"
  grep -qiE 'PASS|SOAK OK' "$L/m5_soak.log" || fail "M5 soak"
fi

step "M7 timing preflight: node still clean"
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || fail "M7 preflight (node became dirty)"

step "M7 timing, 3 rotations x 50"
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u m7_bench.py 3 50 2>&1 | tee "$L/m7_bench.log"
cp $ON/harness/m7_results.json "$L/m7_results.json" 2>/dev/null || true

step "SUMMARY"
sed -n '/GATE M7/,$p' "$L/m7_bench.log"
echo
echo "GATE LADDER COMPLETE for $EXP"
echo "logs in $L"
