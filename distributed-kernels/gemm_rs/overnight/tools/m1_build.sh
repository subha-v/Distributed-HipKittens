#!/usr/bin/env bash
# Gate M1: compile the MI300X/gfx942 GEMM-RS megakernel, production TU and the
# separate negative-control TU. Emits resource-usage diagnostics for Gate M2.
set -uo pipefail

REPO=/home/subvadla/dhk
SRC=$REPO/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
OUT=$REPO/distributed-kernels/gemm_rs/overnight/build
mkdir -p "$OUT"

PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

ROCM_PATH=${ROCM_PATH:-/opt/rocm}

COMMON=(
  -std=c++20 -O3
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math --offload-arch=gfx942
  -I"$REPO/include" -I"$REPO/include/pyutils"
  -I"$ROCM_PATH/include/hip"
  -I"$PBINC" -I"$PYINC"
  -Rpass-analysis=kernel-resource-usage
  -ferror-limit=0 -Wno-nan-infinity-disabled
)

echo "python include: $PYINC"
echo "pybind include: $PBINC"

echo
echo "########## M1a: production TU ##########"
hipcc "${COMMON[@]}" -DTK_MODNAME=gemm_rs_mi300x \
  -c "$SRC" -o "$OUT/gemm_rs_mi300x.o" 2>&1 | tee "$OUT/m1a.log"
echo "M1a exit=${PIPESTATUS[0]}"

echo
echo "########## M1b: negative-control TU ##########"
hipcc "${COMMON[@]}" -DHK_GEMM_RS_MI300X_NEGATIVE_CONTROLS=1 \
  -DTK_MODNAME=gemm_rs_mi300x_control \
  -c "$SRC" -o "$OUT/gemm_rs_mi300x_control.o" 2>&1 | tee "$OUT/m1b.log"
echo "M1b exit=${PIPESTATUS[0]}"

echo
echo "########## summary ##########"
for f in "$OUT/gemm_rs_mi300x.o" "$OUT/gemm_rs_mi300x_control.o"; do
  if [ -f "$f" ]; then echo "OK   $f ($(stat -c%s "$f") bytes)"; else echo "FAIL $f missing"; fi
done
echo "-- first 60 diagnostic lines of m1a --"
grep -E 'error|warning' "$OUT/m1a.log" | head -60
