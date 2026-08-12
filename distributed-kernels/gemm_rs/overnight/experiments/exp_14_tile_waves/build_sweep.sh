#!/usr/bin/env bash
# Build the E4b tile-screening module ONLY.
#
# Separate from harness/build.sh on purpose: the six screening instantiations
# live behind -DHK_GEMM_RS_MI300X_TILE_SWEEP=1 so the production binding keeps
# exactly the six distinct instantiations gate M2 asserts against M2_EXPECT.
# Flags are otherwise byte-identical to build.sh's production arm, so the sweep
# measures the same code generation the graded build gets.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
HARNESS=$GEMM/overnight/harness
OUT=$HARNESS/build
NAME=gemm_rs_mi300x_tilesweep
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

# Force a rebuild every time. exp_ablation.py's "already built" shortcut once
# reported a previous kernel's attribution for an hour; a screening module that
# silently predates the source edit is the same failure with a nicer face.
rm -f "$OUT/$NAME.so"

echo "########## building $NAME (tile sweep rows 101-106) ##########"
hipcc -std=c++20 -O3 \
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 -shared -fPIC \
  -I"$REPO/include" -I"$REPO/include/pyutils" \
  -I"$ROCM_PATH/include/hip" -I"$PBINC" -I"$PYINC" \
  -Wno-nan-infinity-disabled -ferror-limit=0 \
  -DHK_GEMM_RS_MI300X_TILE_SWEEP=1 \
  -DTK_MODNAME=$NAME \
  -Rpass-analysis=kernel-resource-usage \
  "$GEMM/gemm_rs_mi300x.cpp" -o "$OUT/$NAME.so" 2>&1 | tee "$OUT/$NAME.log" | \
  grep -E 'error|Error' | head -40
status=${PIPESTATUS[0]}

if [ -f "$OUT/$NAME.so" ] && [ "$status" = "0" ]; then
  echo "OK   $NAME.so ($(stat -c%s "$OUT/$NAME.so") bytes)"
else
  echo "FAIL $NAME (exit $status)"
  exit 1
fi

echo
echo "########## per-instantiation resources (from -Rpass-analysis) ##########"
# One line per kernel: VGPRs, AGPRs, LDS, spills, occupancy. The screening rows
# are new template arguments, so a spill that only appears at BK=160 or BN=192
# has to be visible BEFORE any timing is believed.
grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS' \
  "$OUT/$NAME.log" | sed 's/remark: //' | \
  awk '/Function Name/{printf "\n%s\n", $0; next} {print}' | \
  grep -vE '^\s*$' | head -120

echo
echo "########## dispatch coverage ##########"
# The init symbol is PyInit_$NAME, so the spec name must BE $NAME.
python3 - "$OUT/$NAME.so" "$NAME" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("  entry points:", [x for x in dir(mod) if not x.startswith("_")])
PY
echo "TILE SWEEP MODULE BUILT"
