#!/usr/bin/env bash
# Post-landing confirmation, after the screening dispatch rows were removed:
#   1. the static-check suite (all ~20 gates, including gate 16's arm count and
#      gate 2-14's derivation of the new tile geometry from the header itself);
#   2. a rebuild of the production modules from the now-clean source;
#   3. M2 again, to prove the instantiation count is still 7 with the screening
#      rows gone -- they were behind a macro, so it must be, and "must be" is
#      exactly the kind of claim that has silently failed in this tree before;
#   4. M3 on the three retiled shapes, so the landed binary is the verified one.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
L=$D/logs
mkdir -p "$L"

echo "########## static checks ##########"
docker exec -w $ON/.. dhk-gemmrs python3 -u gemm_rs_mi300x_static_checks.py \
  2>&1 | tail -30 | tee "$L/static_checks.log"

echo
echo "########## rebuild production modules ##########"
docker exec dhk-gemmrs bash $ON/harness/build.sh 2>&1 | \
  tee "$L/confirm_build.log" | grep -E 'OK |FAIL|ALL MODULES'

echo
echo "########## M2 instantiation count ##########"
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$L/confirm_m1.log" 2>&1
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh  > "$L/confirm_isa.log" 2>&1
docker exec dhk-gemmrs bash $ON/tools/m2_report.sh 2>&1 | \
  tee "$L/confirm_m2.log" | sed -n '/metadata table/,/ordering ops/p'
rows=$(sed -n '/metadata table/,/ordering ops/p' "$L/confirm_m2.log" | \
       grep -cE 'tail=[01]')
echo "  distinct instantiations: $rows (M2_EXPECT is 7)"

echo
echo "########## M3 on the three retiled shapes ##########"
docker exec -w $ON/harness dhk-gemmrs timeout 900 \
  python3 -u m3_correctness.py scored 2>&1 | tail -20 | \
  tee "$L/confirm_m3.log"
echo "CONFIRM DONE"
