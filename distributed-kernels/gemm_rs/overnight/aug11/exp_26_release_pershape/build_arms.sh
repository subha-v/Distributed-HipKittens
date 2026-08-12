#!/usr/bin/env bash
# CPU-only. Build the exp_26 release-group arms as SEPARATE modules from the one
# source, so ab_pershape.py can hold them all open in one process and interleave
# their timed blocks.
#
# Arms:
#   ps0   FULL_ONLY=1 PERSHAPE=0 RG=4   the shipped incumbent rule
#   ps1   FULL_ONLY=1 PERSHAPE=1 RG=4   the candidate, rgroup = min(4, tiles/CTA)
#   rg2c  FULL_ONLY=1 PERSHAPE=0 RG=2   mechanism cross-check: on 8192x4096x14336
#                                       (2 tiles/CTA) this computes rgroup = 2 by
#                                       a different expression than ps1 does, so
#                                       agreement between them on that shape is
#                                       evidence the candidate's rgroup really is
#                                       2 and not a codegen artefact
#   ps0b  FULL_ONLY=1 PERSHAPE=0 RG=4   NULL ARM: a second build of ps0 under a
#                                       different module name. Nothing an
#                                       instruction can see separates it from
#                                       ps0, so whatever the A/B reports between
#                                       them is the instrument's own bias and it
#                                       is the floor under which no other arm's
#                                       delta means anything.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
OUT=$GEMM/overnight/harness/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

spec_flags() {
  case "$1" in
    ps0|ps0b) echo "4 1 0" ;;
    ps1)      echo "4 1 1" ;;
    ps2)      echo "4 1 2" ;;
    rg2c)     echo "2 1 0" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

fail=0
for spec in ${@:-ps0 ps1 ps2 rg2c ps0b}; do
  read -r rg full pershape <<<"$(spec_flags "$spec")" || { fail=1; continue; }
  name=gemm_rs_mi300x_$spec
  echo "########## building $name (RG=$rg FULL_ONLY=$full PERSHAPE=$pershape) ##########"
  # Force a rebuild: a stale .so with a fresh mtime is the exact trap the ledger
  # records (push.ps1 rewrites source mtimes, so mtime is never a freshness
  # test). Removing the target makes "the file exists" a real build result.
  rm -f "$OUT/$name.so"
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
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP=$rg \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=$full \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=$pershape \
    -DTK_MODNAME=$name \
    "$GEMM/gemm_rs_mi300x.cpp" -o "$OUT/$name.so" 2>&1 \
    | tee "$OUT/$name.log" | grep -E 'error|Error' | head -20
  status=${PIPESTATUS[0]}
  if [ -f "$OUT/$name.so" ] && [ "$status" = "0" ]; then
    echo "OK   $name.so ($(stat -c%s "$OUT/$name.so") bytes)"
  else
    echo "FAIL $name (exit $status)"; fail=1
  fi
done

echo
echo "########## arm resource tuples (VGPR / scratch / spill must not move) ##########"
for spec in ${@:-ps0 ps1 ps2 rg2c ps0b}; do
  echo "-- arm $spec --"
  # Anchor on `remark: ` rather than on the start of the line: the -Rpass lines
  # carry a file:line prefix, so `^ *VGPRs:` silently matched nothing and the
  # tuple printed only the scratch and spill halves.
  grep -E 'remark: +(VGPRs|AGPRs|ScratchSize|VGPRs Spill|Occupancy)' \
    "$OUT/gemm_rs_mi300x_$spec.log" \
    | sed 's/.*remark: //; s/ \[-Rpass.*//; s/^ *//' | paste -sd' ' - \
    | sed 's/VGPRs Spill: /sp/g; s/VGPRs: /V/g; s/AGPRs: /A/g; s/ScratchSize \[bytes\/lane\]: /scr/g; s/Occupancy \[waves\/SIMD\]: /occ/g'
done

echo
echo "########## .so sha256 ##########"
for spec in ${@:-ps0 ps1 ps2 rg2c ps0b}; do
  sha256sum "$OUT/gemm_rs_mi300x_$spec.so" 2>/dev/null | sed 's#'"$OUT"'/##'
done

echo
[ "$fail" = "0" ] && echo "ARM MODULES BUILT" || echo "ARM BUILD FAILURES"
exit $fail
