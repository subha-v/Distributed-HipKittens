#!/usr/bin/env bash
# Build the frozen pre-E3 kernel as a SEPARATE module, so m9_stale_slot can hold
# both it and the candidate open in one process and compare their outputs
# bitwise over an identical input sequence.
#
# baseline/gemm_rs_mi300x_e3base.cpp is a byte copy of gemm_rs_mi300x.cpp taken
# immediately before the E3 restructure (sha256 recorded in result.md). It is
# never edited. Flags are build.sh's COMMON verbatim plus -I$GEMM, which the
# in-tree build does not need because its source sits next to the headers.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
HARNESS=$GEMM/overnight/harness
EXP=$GEMM/overnight/experiments/exp_05_release_granularity
SRC=$EXP/baseline/gemm_rs_mi300x_e3base.cpp
OUT=$HARNESS/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

echo "golden source: $(sha256sum "$SRC")"
hipcc \
  -std=c++20 -O3 \
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 \
  -shared -fPIC \
  -I"$REPO/include" -I"$REPO/include/pyutils" -I"$GEMM" \
  -I"$ROCM_PATH/include/hip" \
  -I"$PBINC" -I"$PYINC" \
  -Wno-nan-infinity-disabled -ferror-limit=0 \
  -Rpass-analysis=kernel-resource-usage \
  -DTK_MODNAME=gemm_rs_mi300x_e3base \
  "$SRC" -o "$OUT/gemm_rs_mi300x_e3base.so" 2>&1 | tee "$OUT/gemm_rs_mi300x_e3base.log" | \
  grep -E 'error|Error' | head -30
status=${PIPESTATUS[0]}

if [ -f "$OUT/gemm_rs_mi300x_e3base.so" ] && [ "$status" = "0" ]; then
  echo "OK   gemm_rs_mi300x_e3base.so ($(stat -c%s "$OUT/gemm_rs_mi300x_e3base.so") bytes)"
  python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('gemm_rs_mi300x_e3base',
                                              '$OUT/gemm_rs_mi300x_e3base.so')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('  entry points:', [x for x in dir(mod) if not x.startswith('_')])
"
  echo
  echo "########## golden resource tuples (the C3 baseline) ##########"
  grep -A11 'Function Name' "$OUT/gemm_rs_mi300x_e3base.log" \
    | sed 's/.*remark: //; s/ \[-Rpass.*//' | grep -vE '^--$|^/home' \
    | sed 's/^ *//'
  echo "GOLDEN MODULE BUILT"
  exit 0
fi
echo "FAIL gemm_rs_mi300x_e3base (exit $status)"
exit 1
