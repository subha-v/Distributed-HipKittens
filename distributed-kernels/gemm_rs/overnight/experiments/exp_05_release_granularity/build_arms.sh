#!/usr/bin/env bash
# CPU-only. Build the RELEASE_GROUP arms as three SEPARATE modules from the one
# source, so ab_release_group.py can hold them all open in one process and
# interleave their timed blocks.
#
# This is the only denominator that can resolve this experiment: the ladder's
# per-arm M7 numbers are cross-run, the node's run-to-run spread is ~1.3%, and
# the effect on four of six shapes is smaller than that. Interleaving arm blocks
# inside one process shares the clock state, the allocation, and the input
# sequence, so the only thing that differs between two adjacent samples is the
# release granularity.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
OUT=$GEMM/overnight/harness/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

# Arm specs:
#   N    RELEASE_GROUP = N, grouped unconditionally
#   Nc   RELEASE_GROUP = N, grouped only where a CTA owns N tiles (FULL_ONLY)
#   1b   a SECOND build of RELEASE_GROUP = 1 under a different module name
#
# `1b` is the null arm: it differs from rg1 in nothing an instruction can see,
# so whatever separates them in the A/B is the harness's own positional bias --
# allocation order, which heap a shape's arm got, arm order inside a round --
# and it bounds what any real arm's delta is allowed to claim.
fail=0
for spec in ${@:-1 2 4 4c 1b}; do
  n=${spec%[bc]}
  full=0
  case $spec in *c) full=1;; esac
  name=gemm_rs_mi300x_rg$spec
  echo "########## building $name (RELEASE_GROUP=$n FULL_ONLY=$full) ##########"
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
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP=$n \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=$full \
    -DTK_MODNAME=$name \
    "$GEMM/gemm_rs_mi300x.cpp" -o "$OUT/$name.so" 2>&1 \
    | tee "$OUT/$name.log" | grep -E 'error|Error' | head -20
  status=${PIPESTATUS[0]}
  if [ -f "$OUT/$name.so" ] && [ "$status" = "0" ]; then
    echo "OK   $name.so ($(stat -c%s "$OUT/$name.so") bytes)"
    # The arm must actually carry the constant it claims to: -D on the command
    # line only wins because the source guards the default with #ifndef.
    grep -c 'RELEASE_GROUP' "$OUT/$name.log" >/dev/null
  else
    echo "FAIL $name (exit $status)"; fail=1
  fi
done

echo
echo "########## arm resource tuples (VGPR / scratch / spill must not move) ##########"
for spec in ${@:-1 2 4 4c 1b}; do
  echo "-- arm rg$spec --"
  grep -E '^ *VGPRs:|ScratchSize|VGPRs Spill' "$OUT/gemm_rs_mi300x_rg$spec.log" \
    | sed 's/.*remark: //; s/ \[-Rpass.*//; s/^ *//' | paste -sd' ' - \
    | sed 's/VGPRs: /V/g; s/ScratchSize \[bytes\/lane\]: /scr/g; s/VGPRs Spill: /sp/g'
done

echo
[ "$fail" = "0" ] && echo "ARM MODULES BUILT" || echo "ARM BUILD FAILURES"
exit $fail
