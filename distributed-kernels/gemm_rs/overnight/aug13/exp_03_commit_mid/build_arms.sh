#!/usr/bin/env bash
# exp_03 M1: build the three A/B arms. Runs INSIDE dhk-gemmrs (or via
# docker exec). Never touches build/gemm_rs_mi300x.so.
#   gemm_rs_mi300x_cmid    COMMIT_MID=1   the candidate
#   gemm_rs_mi300x_cmid0   COMMIT_MID=0   the incumbent, same source
#   gemm_rs_mi300x_cmid0b  COMMIT_MID=0   second build of the incumbent (null)
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
OUT=$REPO/distributed-kernels/gemm_rs/overnight/harness/build
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
  -Rpass-analysis=kernel-resource-usage
)

fail=0
build() {
  local name="$1"; shift
  echo "########## building $name ##########"
  hipcc "${COMMON[@]}" "$@" -o "$OUT/$name.so" 2>&1 | tee "$OUT/$name.log" | \
    grep -E 'error|Error' | head -20
  local status=${PIPESTATUS[0]}
  if [ -f "$OUT/$name.so" ] && [ "$status" = "0" ]; then
    echo "OK   $name.so ($(stat -c%s "$OUT/$name.so") bytes)"
  else
    echo "FAIL $name (exit $status)"; fail=1
  fi
}

build gemm_rs_mi300x_cmid \
  -DHK_GEMM_RS_MI300X_COMMIT_MID=1 \
  -DTK_MODNAME=gemm_rs_mi300x_cmid \
  "$GEMM/gemm_rs_mi300x.cpp"

build gemm_rs_mi300x_cmid0 \
  -DHK_GEMM_RS_MI300X_COMMIT_MID=0 \
  -DTK_MODNAME=gemm_rs_mi300x_cmid0 \
  "$GEMM/gemm_rs_mi300x.cpp"

build gemm_rs_mi300x_cmid0b \
  -DHK_GEMM_RS_MI300X_COMMIT_MID=0 \
  -DTK_MODNAME=gemm_rs_mi300x_cmid0b \
  "$GEMM/gemm_rs_mi300x.cpp"

exit $fail
