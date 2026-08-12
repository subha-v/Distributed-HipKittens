#!/usr/bin/env bash
# exp_13 final state: rebuild from the source that is actually on the node, then
# prove the binary agrees with it. The ledger records a failed restore that left
# a stale .so under a new source, so "it was built earlier from something
# semantically identical" is not good enough as a closing statement.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_13_cta_split
mkdir -p "$D/logs"

echo "=== node ==="
rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l | sed 's/^/KFD pids: /'

echo
echo "=== the reducer-split column on the node ==="
sed -n '/inline constexpr std::array<shape_entry, 6> scored_shapes/,/}};/p' \
  $ON/../gemm_rs_mi300x_host_abi.hpp
grep -n 'generic_config' $ON/../gemm_rs_mi300x_host_abi.hpp | head -2

echo
echo "=== rebuild ==="
docker exec dhk-gemmrs bash $ON/harness/build.sh > "$D/logs/final_build.log" 2>&1
grep -c 'ALL MODULES BUILT' "$D/logs/final_build.log" | sed 's/^/ALL MODULES BUILT: /'
md5sum $ON/harness/build/gemm_rs_mi300x.so $ON/harness/build/dhk_rt.so

echo
echo "=== resolved plan per graded shape (NR must read 56/32/32/32/32/48) ==="
docker exec -w $ON/harness dhk-gemmrs python3 -u $D/plan_check.py 2>&1

echo
echo "=== M3 correctness on the six graded shapes, both tolerances ==="
docker exec -w $ON/harness dhk-gemmrs timeout 1200 python3 -u m3_correctness.py scored \
  2>&1 | tail -14
