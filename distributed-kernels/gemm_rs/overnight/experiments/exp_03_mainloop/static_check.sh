#!/usr/bin/env bash
# Cheap pre-GPU screen for one arm: compile, resource table, LDS-hazard scan,
# k-loop program order. No GPU. Run this before every ladder -- a hazard here is
# a guaranteed M3 failure and costs 30 s to find instead of 3 min.
set -uo pipefail
ARM=${1:?usage: static_check.sh <arm_name>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
A=$EXP/arms/$ARM
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s
mkdir -p "$A"

echo "########## $ARM: compile + ISA ##########"
docker exec dhk-gemmrs bash $ON/tools/m1_build.sh > "$A/m1_build.log" 2>&1
if grep -qE 'error:' "$A/m1_build.log"; then
  echo "=== COMPILE ERRORS ==="; grep -E 'error:' "$A/m1_build.log" | head -40; exit 1
fi
docker exec dhk-gemmrs bash $ON/tools/m2_isa.sh > "$A/m2_isa.log" 2>&1
[ -f "$S" ] || { echo "NO ISA"; tail -20 "$A/m2_isa.log"; exit 1; }
cp "$S" "$A/gemm_rs_mi300x.gfx942.s"
echo "ok"

echo
bash "$EXP/res_table.sh" "$ARM"

echo
echo "########## $ARM: LDS hazard scan ##########"
bash "$EXP/lds_race_check.sh" "$S" | tee "$A/lds_race.txt" | grep -E 'SYMBOL|clean|HAZARD|reads'

echo
echo "########## $ARM: k-loop program order ##########"
docker exec dhk-gemmrs bash $EXP/p0_30_kloop.sh > "$A/kloop_run.log" 2>&1
cp "$EXP/logs/p0_kloop_inventory.txt" "$A/kloop_inventory.txt" 2>/dev/null || true
sed -n '/DEEPEST MFMA LOOP/,/raw dump/p' "$A/kloop_inventory.txt" 2>/dev/null | \
  grep -E 'SYMBOL|BLOCK|\[A\]|\[C\]|====' | head -70
echo "################ DONE ################"
