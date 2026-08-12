#!/usr/bin/env bash
# exp_22 build, container side. Runs inside subha_k1 (HOME is /root there, so
# every path is absolute).
set -uo pipefail
WANT_ISA=${1:-}
E22=/home/subvadla/e22
DHK=$E22/DHK
OUT=$E22/out
SRC=$E22/src/e22_saturation.hip
mkdir -p "$OUT"

# The megakernel flag contract from BUILDING.md, minus --genco: this is a
# standalone executable so it also needs the host link. -ffast-math and
# -amdgpu-mfma-vgpr-form=1 are kept so the MFMA lowering matches the kernel
# being modelled.
FLAGS=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1
  -Rpass-analysis=kernel-resource-usage
  -I"$DHK/include")

echo "===SRC_IDENTITY==="
sha256sum "$SRC"
grep -m1 'define E22_SRC_REV' "$SRC"
git config --global --add safe.directory "$DHK" 2>/dev/null
git -C "$DHK" rev-parse --short HEAD

echo "===BUILD==="
t0=$SECONDS
hipcc "${FLAGS[@]}" "$SRC" -o "$OUT/e22_saturation" > "$OUT/build.log" 2>&1
rc=$?
echo "exit=$rc secs=$((SECONDS-t0)) errors=$(grep -c 'error:' "$OUT/build.log")"
if [ "$rc" -ne 0 ]; then
  echo "--- diagnostics ---"
  grep -nE 'error:' "$OUT/build.log" | head -40
  exit "$rc"
fi
sha256sum "$OUT/e22_saturation"
ls -l --time-style=+%FT%TZ "$OUT/e22_saturation"

echo "===RESOURCE_REMARK==="
python3 "$E22/src/remark.py" "$OUT/build.log"

if [ "$WANT_ISA" = "isa" ]; then
  echo "===ISA==="
  hipcc "${FLAGS[@]}" --cuda-device-only -S -o "$OUT/e22.s" "$SRC" \
      > "$OUT/isa.log" 2>&1
  echo "asm exit=$? lines=$(wc -l < "$OUT/e22.s" 2>/dev/null || echo 0)"
  python3 "$E22/src/isa_census.py" "$OUT/e22.s"
fi
echo "===DONE==="
