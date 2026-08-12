#!/usr/bin/env bash
# exp_22 GATE 1 -- resource-tuple parity with the trace flag compiled in but OFF.
#
# Three TUs, identical apart from the flag, all compiled with the SAME option
# vector harness/build.sh uses for the production module (that is the point:
# a different -O or a different -D would make the comparison meaningless):
#
#   off_absent   flag not defined at all      = today's production build
#   off_present  -DHK_GEMM_RS_MI300X_TRACE=0  = the build that must be identical
#   on           -DHK_GEMM_RS_MI300X_TRACE=1  = the diagnostic arm, recorded only
#
# PASS requires off_absent and off_present to be byte-identical in
# VGPR/AGPR/SGPR/scratch/spill/LDS across all 7 instantiations, AND both to
# match the post-exp_14 M2 table hard-coded in parity_check.py. The ON tuple is
# recorded for disclosure and is not a pass/fail condition.
#
# CPU-only. Compiling does not touch the GPU and does not need the node lease.
# Artifacts land in exp_22_timeline/build/ -- never in harness/build/.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
SRC=$GEMM/gemm_rs_mi300x.cpp
OUT=$GEMM/overnight/aug11/exp_22_timeline/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

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

compile() {   # compile <tag> [extra -D...]
  local tag=$1; shift
  echo "########## $tag ##########"
  hipcc "${COMMON[@]}" -DTK_MODNAME=gemm_rs_mi300x "$@" \
    -c "$SRC" -o "$OUT/$tag.o" 2>&1 | tee "$OUT/$tag.log" >/dev/null
  local rc=${PIPESTATUS[0]}
  if [ -f "$OUT/$tag.o" ] && [ "$rc" = "0" ]; then
    echo "OK   $tag.o ($(stat -c%s "$OUT/$tag.o") bytes)"
  else
    echo "FAIL $tag (exit $rc)"
    grep -E 'error' "$OUT/$tag.log" | head -20
  fi
}

compile off_absent
compile off_present -DHK_GEMM_RS_MI300X_TRACE=0
compile on          -DHK_GEMM_RS_MI300X_TRACE=1

echo
echo "########## verdict ##########"
python3 "$GEMM/overnight/aug11/exp_22_timeline/parity_check.py" \
  --logs "$OUT" --json "$GEMM/overnight/aug11/exp_22_timeline/parity.json"
