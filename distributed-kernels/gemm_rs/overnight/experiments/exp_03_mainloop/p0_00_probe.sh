#!/usr/bin/env bash
# P0 step 0: read-only probe. Does a current ISA artifact exist on the node?
# NO GPU WORK, NO BUILD.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
GEMM=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "=============== host / date ==============="
hostname; date -u '+%Y-%m-%dT%H:%M:%SZ'

echo
echo "=============== GPU owners (read-only, informational) ==============="
rocm-smi --showpids 2>/dev/null | head -20 || echo "(rocm-smi unavailable on host)"

echo
echo "=============== kernel source mtimes ==============="
for f in "$GEMM"/gemm_rs_mi300x.cpp "$GEMM"/gemm_rs_mi300x_hk_adapter.cuh \
         "$GEMM"/gemm_rs_mi300x_constants.cuh "$GEMM"/gemm_rs_mi300x_host_abi.hpp; do
  [ -f "$f" ] && printf '%s  %s  %s\n' "$(stat -c '%y' "$f")" "$(md5sum "$f" | cut -c1-12)" "$f"
done

echo
echo "=============== overnight/build/isa ==============="
ls -la "$ON/build/isa" 2>/dev/null || echo "MISSING: $ON/build/isa"

echo
echo "=============== overnight/build (top) ==============="
ls -la "$ON/build" 2>/dev/null | head -20 || echo "MISSING: $ON/build"

echo
echo "=============== overnight/harness/build ==============="
ls -la "$ON/harness/build" 2>/dev/null | head -20 || echo "MISSING: $ON/harness/build"

echo
echo "=============== any .s anywhere under overnight ==============="
find "$ON" -name '*.s' -o -name '*-gfx942.s' 2>/dev/null | head -20

echo
echo "=============== containers ==============="
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null | head -10
echo "=============== DONE ==============="