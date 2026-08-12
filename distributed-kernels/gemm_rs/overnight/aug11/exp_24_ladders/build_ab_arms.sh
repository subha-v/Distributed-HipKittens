#!/usr/bin/env bash
# exp_27 (under exp_24_ladders): build the three modules the paired ours/ours_prev
# test needs, ALL from the current source, differing only in one macro.
#
#   ab0   PERSHAPE=0  RG=4  FULL_ONLY=1   -> arm `ours_prev` (the pre-exp_26 rule)
#   ab2   PERSHAPE=2  RG=4  FULL_ONLY=1   -> arm `ours`      (the shipped rule)
#   ab2b  PERSHAPE=2  RG=4  FULL_ONLY=1   -> arm `ours_null` (second build of ab2)
#
# Fresh names rather than exp_26's ps0/ps2: those artifacts are from 05:48 and I
# want provenance I can state in one sentence -- three modules, one source tree,
# one differing -D, built minutes before the measurement.
#
# `ours_null` is a SEPARATE .so here, not the same file under a second Python
# name as in the ladder. That is deliberate: `ours_prev` is necessarily a
# separate .so, so the null must be too, or the null would be structurally
# easier than the treatment (shared C++ globals, one dlopen, one IPC exchange)
# and would understate the floor.
#
# Every build is checked for rc AND for the artifact, because a script that dies
# partway can otherwise leave a stale .so that reads as a pass -- that is exactly
# how the CRLF-truncated harness/build.sh printed a "sha differs" line earlier
# tonight for a module it had never written.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
OUT=$GEMM/overnight/harness/build
D=$GEMM/overnight/aug11/exp_24_ladders
mkdir -p "$OUT" "$D/logs"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}

pershape_of() {
  case "$1" in
    ab0)       echo 0 ;;
    ab2|ab2b)  echo 2 ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

fail=0
for spec in ab0 ab2 ab2b; do
  ps=$(pershape_of "$spec") || { fail=1; continue; }
  name=gemm_rs_mi300x_$spec
  echo "########## $name (RG=4 FULL_ONLY=1 PERSHAPE=$ps) ##########"
  rm -f "$OUT/$name.so"
  docker exec dhk-gemmrs bash -lc "
    hipcc -std=c++20 -O3 \
      -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
      -ffast-math --offload-arch=gfx942 -shared -fPIC \
      -I$REPO/include -I$REPO/include/pyutils -I$GEMM \
      -I\$ROCM_PATH/include/hip \
      -I/usr/local/lib/python3.12/dist-packages/pybind11/include \
      -I/usr/include/python3.12 \
      -Wno-nan-infinity-disabled -ferror-limit=0 \
      -Rpass-analysis=kernel-resource-usage \
      -DHK_GEMM_RS_MI300X_RELEASE_GROUP=4 \
      -DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=1 \
      -DHK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=$ps \
      -DTK_MODNAME=$name \
      $GEMM/gemm_rs_mi300x.cpp -o $OUT/$name.so" \
    > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$OUT/$name.so" ]; then
    echo "  FAIL rc=$rc artifact=$([ -f "$OUT/$name.so" ] && echo yes || echo no)"
    grep -E 'error' "$OUT/$name.log" | head -8
    fail=1
    continue
  fi
  echo "  OK   $(stat -c%s "$OUT/$name.so") B  sha $(sha256sum "$OUT/$name.so" | cut -c1-16)"
  echo "  init symbol: $(nm -D --defined-only "$OUT/$name.so" | grep -o "PyInit_$name" | head -1)"
done
[ $fail -ne 0 ] && { echo "FATAL: not all arms built"; exit 1; }

echo
echo "########## the arms must be distinguishable in the right way ##########"
S0=$(sha256sum "$OUT/gemm_rs_mi300x_ab0.so" | cut -d' ' -f1)
S2=$(sha256sum "$OUT/gemm_rs_mi300x_ab2.so" | cut -d' ' -f1)
Z0=$(stat -c%s "$OUT/gemm_rs_mi300x_ab0.so")
Z2=$(stat -c%s "$OUT/gemm_rs_mi300x_ab2.so")
echo "  ab0  ${S0:0:16}  $Z0 B"
echo "  ab2  ${S2:0:16}  $Z2 B"
echo "  ab2b $(sha256sum "$OUT/gemm_rs_mi300x_ab2b.so" | cut -c1-16)  $(stat -c%s "$OUT/gemm_rs_mi300x_ab2b.so") B"
# ab0 and ab2 have module names of equal length, so a size/content difference is
# attributable to the macro and nothing else. If they were identical the two arms
# would be the same computation and the whole test would be void.
if [ "$S0" = "$S2" ]; then
  echo "  FATAL: ab0 and ab2 are byte-identical -- PERSHAPE never reached codegen,"
  echo "         so there is no A/B to run."
  exit 1
fi
echo "  OK: ab0 != ab2, so the two arms really are different code"

echo
echo "########## resource tuples (must not move between the arms) ##########"
for spec in ab0 ab2 ab2b; do
  lg="$OUT/gemm_rs_mi300x_$spec.log"
  echo "-- $spec ($(wc -l < "$lg") log lines) --"
  grep -oE '(VGPRs|AGPRs|SGPRs|ScratchSize|Occupancy \[waves/SIMD\]|Spills?)[^,]*' "$lg" \
    | sed 's/  */ /g' | sort | uniq -c | sort -rn | head -10 | sed 's/^/    /'
done

echo
{
  echo "ab0  PERSHAPE=0  $S0"
  echo "ab2  PERSHAPE=2  $S2"
  echo "ab2b PERSHAPE=2  $(sha256sum "$OUT/gemm_rs_mi300x_ab2b.so" | cut -d' ' -f1)"
} > "$D/logs/ab_arm_shas.txt"
cat "$D/logs/ab_arm_shas.txt"
echo "########## BUILD COMPLETE ##########"
