#!/usr/bin/env bash
# exp_22 Phase 2: build the flag-ON diagnostic module.
#
# Same option vector as harness/build.sh's production module plus
# -DHK_GEMM_RS_MI300X_TRACE=1, and a distinct TK_MODNAME so the .so can never
# be mistaken for the production one.  The exported entry point is still
# `gemm_rs_mi300x` (bind_function's name is a literal in the source), so
# harness_lib's calling convention applies unchanged -- only the module file
# name and the extra trailing argument differ.
#
# Output goes to exp_22_timeline/build/, NEVER harness/build/: the campaign
# harness must not be able to load a diagnostic build by accident.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
OUT=$GEMM/overnight/aug11/exp_22_timeline/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

hipcc -std=c++20 -O3 \
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 \
  -shared -fPIC \
  -I"$REPO/include" -I"$REPO/include/pyutils" \
  -I"$ROCM_PATH/include/hip" -I"$PBINC" -I"$PYINC" \
  -Wno-nan-infinity-disabled -ferror-limit=0 \
  -Rpass-analysis=kernel-resource-usage \
  -DHK_GEMM_RS_MI300X_TRACE=1 \
  -DTK_MODNAME=gemm_rs_mi300x_trace \
  "$GEMM/gemm_rs_mi300x.cpp" -o "$OUT/gemm_rs_mi300x_trace.so" \
  2>&1 | tee "$OUT/trace_build.log" | grep -E 'error|Error' | head -30
rc=${PIPESTATUS[0]}

if [ -f "$OUT/gemm_rs_mi300x_trace.so" ] && [ "$rc" = "0" ]; then
  echo "OK   gemm_rs_mi300x_trace.so ($(stat -c%s "$OUT/gemm_rs_mi300x_trace.so") bytes)"
  # The spec name is not cosmetic: CPython resolves a C extension's init symbol
  # as PyInit_<last component of the spec name>, so loading this .so under any
  # name other than TK_MODNAME fails with a misleading "does not define module
  # export function" that reads like a build failure and is not one.
  python3 -c "
import importlib.util
name = 'gemm_rs_mi300x_trace'
spec = importlib.util.spec_from_file_location(name, '$OUT/' + name + '.so')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print('  entry points:', [x for x in dir(mod) if not x.startswith('_')])"
else
  echo "FAIL trace build (exit $rc)"; exit 1
fi
