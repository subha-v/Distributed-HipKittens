#!/usr/bin/env bash
# rank-1's C launcher parses a 6-field packed_metadata; Triton 3.6.0 supplies a
# different number. Establish the 3.6.0 layout and how rank-1 uses each field,
# to size a faithful repair against simply disabling their launcher.
set -uo pipefail
NAME=dhk-eval
R=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
D=/usr/local/lib/python3.12/dist-packages/triton
run() { docker exec "$NAME" bash -c "$1"; }

echo "===== triton 3.6.0: what is packed_metadata? ====="
run "grep -rn 'packed_metadata' $D/compiler/compiler.py $D/backends/amd/compiler.py 2>/dev/null | head -20"
echo "--- its construction ---"
run "grep -n -B6 -A12 'packed_metadata' $D/backends/amd/compiler.py 2>/dev/null | head -40"

echo
echo "===== triton 3.6.0 stock launcher: how it parses kernel_metadata ====="
run "grep -n -B3 -A6 'PyArg_ParseTuple(kernel_metadata' $D/backends/amd/driver.py | head -30"

echo
echo "===== rank-1: how its C launcher parses and uses kernel_metadata ====="
grep -n -B6 -A30 'PyArg_ParseTuple(kernel_metadata' "$R" | head -60

echo
echo "===== rank-1: does its C code use clusterDim / num_ctas at all? ====="
grep -n 'clusterDim\|num_ctas\|num_warps\|shared_memory' "$R" | head -30

echo
echo "===== rank-1: the _BASE_ARGS_FORMAT it relies on ====="
run "grep -n '_BASE_ARGS_FORMAT' $D/backends/amd/driver.py | head"
grep -n '_BASE_ARGS_FORMAT\|format =' "$R" | head -15
