#!/usr/bin/env bash
# Build the three modules the validation harness needs:
#   gemm_rs_mi300x          production kernel binding (one entry point)
#   gemm_rs_mi300x_control  negative-control binding (separate TU product)
#   dhk_rt                  symmetric-heap / descriptor / shape-resolver glue
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
HARNESS=$REPO/.node/harness
OUT=$HARNESS/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

COMMON=(
  -std=c++20 -O3
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math --offload-arch=gfx942
  -shared -fPIC
  -I"$REPO/include" -I"$REPO/include/pyutils"
  -I"$ROCM_PATH/include/hip"
  -I"$PBINC" -I"$PYINC"
  -Wno-nan-infinity-disabled -ferror-limit=0
)

fail=0

build() {
  local name="$1"; shift
  echo "########## building $name ##########"
  hipcc "${COMMON[@]}" "$@" -o "$OUT/$name.so" 2>&1 | tee "$OUT/$name.log" | \
    grep -E 'error|Error' | head -30
  local status=${PIPESTATUS[0]}
  if [ -f "$OUT/$name.so" ] && [ "$status" = "0" ]; then
    echo "OK   $name.so ($(stat -c%s "$OUT/$name.so") bytes)"
  else
    echo "FAIL $name (exit $status)"; fail=1
  fi
}

build gemm_rs_mi300x \
  -DTK_MODNAME=gemm_rs_mi300x \
  -Rpass-analysis=kernel-resource-usage \
  "$GEMM/gemm_rs_mi300x.cpp"

build gemm_rs_mi300x_control \
  -DHK_GEMM_RS_MI300X_NEGATIVE_CONTROLS=1 \
  -DTK_MODNAME=gemm_rs_mi300x_control \
  "$GEMM/gemm_rs_mi300x.cpp"

build dhk_rt \
  -I"$GEMM" -I"$HARNESS/shim" \
  "$HARNESS/dhk_rt.cpp"

echo
echo "########## exported symbols ##########"
for m in gemm_rs_mi300x gemm_rs_mi300x_control dhk_rt; do
  if [ -f "$OUT/$m.so" ]; then
    echo "-- $m --"
    python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('$m', '$OUT/$m.so')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('  entry points:', [x for x in dir(mod) if not x.startswith('_')])
" 2>&1 | tail -3
  fi
done

echo
[ "$fail" = "0" ] && echo "ALL MODULES BUILT" || echo "BUILD FAILURES PRESENT"
exit $fail
