#!/usr/bin/env bash
# CPU-only build/inspection entry point. This script never launches a kernel.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${DHK_ROOT:-$(git -C "$HERE" rev-parse --show-toplevel)}
OUT=${OUT_DIR:-"$HERE/build"}
SRC=${SRC_FILE:-"$HERE/neutral_transport.hip"}
ARCH=${GPU_ARCH:-gfx950}
WANT_ISA=0

if [[ ${1:-} == "--isa" || ${1:-} == "isa" ]]; then
  WANT_ISA=1
elif [[ $# -ne 0 ]]; then
  echo "usage: bash build.sh [--isa]" >&2
  exit 2
fi

mkdir -p "$OUT"
FLAGS=(
  "--offload-arch=$ARCH"
  -std=c++20
  -O3
  -DKITTENS_CDNA4
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -Rpass-analysis=kernel-resource-usage
  -I"$ROOT/include"
)

echo "=== source identity ==="
sha256sum "$SRC"
awk '/define NEUTRAL_TRANSPORT_SRC_REV/{print; exit}' "$SRC"
if ! git -C "$ROOT" branch --show-current 2>/dev/null; then
  echo "branch=unavailable (read-only bind ownership)"
fi
if ! git -C "$ROOT" rev-parse HEAD 2>/dev/null; then
  echo "head=unavailable (read-only bind ownership)"
fi
printf 'flags='
printf ' %q' "${FLAGS[@]}"
printf '\n'

echo "=== executable build (CPU compile/link only) ==="
hipcc "${FLAGS[@]}" "$SRC" -o "$OUT/neutral_transport" \
  >"$OUT/build.log" 2>&1
sha256sum "$OUT/neutral_transport"
stat -c 'binary=%n bytes=%s mtime=%y' "$OUT/neutral_transport"

echo "=== kernel resource remarks ==="
if command -v rg >/dev/null 2>&1; then
  RESOURCE_MATCH=(rg -n "remark:|Function Name:|VGPR|SGPR|AGPR|Scratch|LDS|Occupancy")
else
  RESOURCE_MATCH=(awk '/remark:|Function Name:|VGPR|SGPR|AGPR|Scratch|LDS|Occupancy/
    {printf "%d:%s\n", NR, $0; found=1} END {exit found ? 0 : 1}')
fi
if ! "${RESOURCE_MATCH[@]}" "$OUT/build.log"; then
  echo "WARNING: no kernel-resource remark matched; inspect $OUT/build.log"
fi

if [[ $WANT_ISA -eq 1 ]]; then
  echo "=== device assembly ==="
  hipcc "${FLAGS[@]}" --cuda-device-only -S "$SRC" \
    -o "$OUT/neutral_transport.s" >"$OUT/isa.log" 2>&1
  stat -c 'assembly=%n bytes=%s mtime=%y' "$OUT/neutral_transport.s"
  echo "=== ISA concern census ==="
  for opcode in \
      flat_load_dwordx4 flat_store_dwordx4 buffer_wbl2 \
      s_waitcnt s_barrier global_atomic_add scratch_load scratch_store; do
    if command -v rg >/dev/null 2>&1; then
      count=$(rg -c "$opcode" "$OUT/neutral_transport.s" || true)
    else
      count=$(awk -v opcode="$opcode" \
        'index($0, opcode) {count++} END {print count + 0}' \
        "$OUT/neutral_transport.s")
    fi
    printf '%-24s %s\n' "$opcode" "${count:-0}"
  done
fi

echo "BUILD_CPU_ONLY: PASS"
