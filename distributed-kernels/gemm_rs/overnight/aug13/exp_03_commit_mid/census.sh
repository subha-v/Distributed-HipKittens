#!/usr/bin/env bash
# exp_03 M2: CPU-only ISA census of both arms + the placement assert.
# Mirrors exp_27/isa_census.sh; writes only under exp_03_commit_mid/isa/.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
EXP=$GEMM/overnight/aug13/exp_03_commit_mid
EXP27=$GEMM/overnight/aug11/exp_27_mainloop
ISA=$EXP/isa
mkdir -p "$ISA"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

echo "################ provenance ################"
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh \
         gemm_rs_mi300x_constants.cuh; do
  printf '%-40s %s\n' "$f" "$(sha256sum "$GEMM/$f" | cut -c1-16)"
done
echo "utc $(date -u +%Y%m%dT%H%M%SZ)"

compile_arm() {  # compile_arm <tag> <flag_value>
  local tag=$1 flag=$2
  local dir=$ISA/$tag
  mkdir -p "$dir"; cd "$dir" || exit 1
  rm -f ./*.s ./*.o ./*.bc ./*.hipi ./*.cui 2>/dev/null
  echo "---- compiling $tag (COMMIT_MID=$flag) ----"
  hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
    -ffast-math --offload-arch=gfx942 -DTK_MODNAME=gemm_rs_mi300x \
    -DHK_GEMM_RS_MI300X_COMMIT_MID=$flag \
    -I"$REPO/include" -I"$REPO/include/pyutils" -I"$ROCM_PATH/include/hip" \
    -I"$PBINC" -I"$PYINC" -Wno-nan-infinity-disabled -ferror-limit=0 \
    -Rpass-analysis=kernel-resource-usage \
    --save-temps -c "$GEMM/gemm_rs_mi300x.cpp" -o "$dir/arm.o" \
    >"$dir/compile.log" 2>&1
  echo "exit=$?"
  local S
  S=$(ls "$dir"/*gfx942*.s 2>/dev/null | head -1)
  [ -z "$S" ] && S=$(ls "$dir"/*.s 2>/dev/null | head -1)
  [ -z "$S" ] && { echo "NO ISA FILE for $tag - abort"; exit 1; }
  cp "$S" "$dir/kernel.s"
  python3 "$EXP27/kloop_hist.py" "$dir/kernel.s" "$dir/kloop.json" || true
  rm -f "$dir"/*.bc "$dir"/*.hipi "$dir"/*.cui "$dir"/*.o 2>/dev/null
}

compile_arm base 0
compile_arm cmid 1

echo
echo "################ default-off ratchet vs exp_27 archive ################"
OLD=$(ls "$EXP27"/isa/*gfx942*.s 2>/dev/null | head -1)
if [ -n "$OLD" ]; then
  if diff -q "$OLD" "$ISA/base/kernel.s" >/dev/null 2>&1; then
    echo "RATCHET OK: flag-off ISA identical to exp_27 census"
  else
    echo "RATCHET DIFF: flag-off ISA differs from exp_27 census ($(diff "$OLD" "$ISA/base/kernel.s" | wc -l) diff lines)"
    echo "  (exp_27 censused the pre-revert source; investigate before trusting)"
  fi
else
  echo "exp_27 archived .s not present on node; ratchet check SKIPPED"
fi

echo
echo "################ placement assert ################"
python3 "$EXP/assert_placement.py" "$ISA/base/kernel.s" "$ISA/cmid/kernel.s" \
  "$ISA/base/compile.log" "$ISA/cmid/compile.log"
