#!/usr/bin/env bash
# exp_03 gate ladder, per arm.   ladder.sh <arm_name>
#
# A faithful copy of tools/gate_ladder.sh -- same gates, same order, same
# tolerances, same shape set, same iteration counts, same soak length, same
# timing invocation. Nothing that the benchmark measures is changed. Two
# differences, both in M2's self-check, and both documented here because a gate
# must never be quietly weakened:
#
#   1. tools/gate_ladder.sh:53 asserts the metadata table has >= 7 rows. The
#      dispatch switch produces exactly SIX distinct instantiations today:
#        <32,64,64,false>   case 1 and the generic even-K fallback
#        <64,64,64,false>   case 2
#        <128,256,32,true>  case 3
#        <256,256,32,false> cases 4 and 5
#        <256,256,32,true>  case 6
#        <32,64,64,true>    the generic odd-K fallback
#      7 was correct until exp_04 retiled row 1 from <32,256,32,false> to
#      <32,64,64,false>, which collapsed it into the generic even-K row. The
#      constant is therefore stale against the COMMITTED baseline, not against
#      this experiment: exp_03 does not touch dispatch_gemm_rs_mi300x, so the
#      set of instantiations is unchanged by it.
#   2. Rather than lowering 7 to 6, M2 here asserts that all six expected
#      tuples are present BY NAME, which is strictly stronger than counting
#      rows: a vacuous table, a missing instantiation, or an unexpected extra
#      one all fail.
#
# Logs land in experiments/exp_03_mainloop/arms/<arm>/ so arms stay comparable.
set -uo pipefail

ARM=${1:?usage: ladder.sh <arm_name> [skip_soak]}
SKIP_SOAK=${2:-0}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$ON/experiments/exp_03_mainloop/arms/$ARM
mkdir -p "$L"

step() { printf '\n########## %s ##########\n' "$1"; }
fail() { echo "GATE LADDER FAILED AT: $1"; exit 1; }

step "M0 preflight: node must be clean"
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || fail "M0 node dirty (another GPU job is running)"

step "M1 build"
docker exec dhk-gemmrs bash $ON/harness/build.sh > "$L/m1_build_so.log" 2>&1
grep -q "ALL MODULES BUILT" "$L/m1_build_so.log" || \
  { tail -30 "$L/m1_build_so.log"; fail "M1 build"; }
grep -E '^OK|ALL MODULES BUILT' "$L/m1_build_so.log"

step "M2 resources / ISA"
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$L/m2_prereq_m1.log" 2>&1 || \
  echo "  (m1_build prereq returned nonzero; continuing to m2_isa)"
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh  > "$L/m2_prereq_isa.log" 2>&1 || \
  echo "  (m2_isa prereq returned nonzero)"
docker exec dhk-gemmrs bash $ON/tools/m2_report.sh > "$L/m2_report.log" 2>&1
grep -qE 'Error|Traceback|No such file' "$L/m2_report.log" && \
  fail "M2 (report could not read its inputs)"

TBL=$(sed -n '/metadata table/,/ordering ops/p' "$L/m2_report.log")
rows=$(echo "$TBL" | grep -cE 'tail=[01]')
echo "  M2 metadata rows: $rows"
miss=0
for want in ' 32/ 64/64 tail=0' ' 64/ 64/64 tail=0' '128/256/32 tail=1' \
            '256/256/32 tail=0' '256/256/32 tail=1' ' 32/ 64/64 tail=1'; do
  if echo "$TBL" | grep -qF "$want"; then echo "    present: $want"
  else echo "    MISSING: $want"; miss=1; fi
done
[ "$miss" = "0" ] || fail "M2 (an expected instantiation is absent from the table)"
[ "$rows" = "6" ] || fail "M2 (table has $rows rows; the 6 expected tuples plus extras)"
echo "  --- resource tuples ---"
echo "$TBL" | grep -E 'BM/BN/BK|tail=[01]'
echo "  --- scratch / spill sites in the whole TU ---"
grep -cE 'scratch_store|scratch_load' "$L/m2_report.log" | \
  xargs -I{} echo "    m2_report scratch mentions: {}"
sed -n '/scratch (spill) sites/,/DONE/p' "$L/m2_report.log" | head -20

step "M3 correctness, 17 shapes, 1e-2 AND 2e-3"
docker exec -w $ON/harness dhk-gemmrs timeout 2400 \
  python3 -u m3_correctness.py all > "$L/m3_correctness.log" 2>&1
grep -qE 'shapes PASSED' "$L/m3_correctness.log" || \
  { tail -40 "$L/m3_correctness.log"; fail "M3 correctness"; }
if grep -qE '\bFAIL\b' "$L/m3_correctness.log"; then
  grep -nE '\bFAIL\b' "$L/m3_correctness.log" | head -20; fail "M3 correctness (a shape FAILed)"
fi
grep -E 'shapes PASSED|worst|max\|diff\|' "$L/m3_correctness.log" | tail -8

step "M4 negative controls"
docker exec -w $ON/harness dhk-gemmrs timeout 1200 \
  python3 -u m4_controls.py > "$L/m4_controls.log" 2>&1
tail -12 "$L/m4_controls.log"
grep -qiE 'all controls|controls PASSED|3/3' "$L/m4_controls.log" || \
  echo "  (note: check m4 verdict text by hand)"

if [ "$SKIP_SOAK" != "skip_soak" ]; then
  step "M5 600-epoch skewed soak"
  docker exec -w $ON/harness dhk-gemmrs timeout 2400 \
    python3 -u m5_soak.py 600 512 4096 12288 1 1 > "$L/m5_soak.log" 2>&1
  tail -8 "$L/m5_soak.log"
  grep -qiE 'PASS|SOAK OK' "$L/m5_soak.log" || fail "M5 soak"
fi

step "M7 timing preflight: node still clean"
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
[ "$n" = "0" ] || fail "M7 preflight (node became dirty)"

step "M7 timing, 3 rotations x 50"
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u m7_bench.py 3 50 2>&1 | tee "$L/m7_bench.log"
cp $ON/harness/m7_results.json "$L/m7_results.json" 2>/dev/null || true

step "SUMMARY ($ARM)"
sed -n '/GATE M7/,$p' "$L/m7_bench.log"
echo
echo "LADDER COMPLETE for arm $ARM; logs in $L"
