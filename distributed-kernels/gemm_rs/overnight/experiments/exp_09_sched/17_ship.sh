#!/usr/bin/env bash
# Put the winning arm back in place and prove the shipped state end to end.
#   17_ship.sh <arm>
set -uo pipefail
ARM=${1:?arm}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
SRC=$ON/..

echo "########## restore $ARM ##########"
find "$SRC" -maxdepth 1 -type f -writable \( -name '*.cpp' -o -name '*.cuh' -o -name '*.hpp' \) \
  -exec sed -i 's/\r$//' {} +
cp -f "$EXP/arms/$ARM/gemm_rs_mi300x.cpp"            "$SRC/gemm_rs_mi300x.cpp"
cp -f "$EXP/arms/$ARM/gemm_rs_mi300x_hk_adapter.cuh" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"
sha256sum "$SRC/gemm_rs_mi300x.cpp" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"

echo
bash "$EXP/run_ladder.sh" exp_09_sched
rc=$?

mkdir -p "$EXP/arms/$ARM/gates_confirm"
cp -f "$EXP/logs/"m*.log "$EXP/arms/$ARM/gates_confirm/" 2>/dev/null || true
cp -f "$EXP/logs/m7_results.json" "$EXP/arms/$ARM/gates_confirm/" 2>/dev/null || true

echo
echo "########## SHIPPED STATE ##########"
sha256sum "$SRC/gemm_rs_mi300x.cpp" "$SRC/gemm_rs_mi300x_hk_adapter.cuh"
sed -n '/BM\/BN\/BK/,/^$/p' "$EXP/logs/m2_report.log" | head -8
sed -n '/GATE M7/,$p' "$EXP/logs/m7_bench.log" | head -20
echo "17_ship rc=$rc"
exit $rc
