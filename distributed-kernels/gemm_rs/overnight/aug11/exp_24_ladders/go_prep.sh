#!/usr/bin/env bash
# exp_24 re-measure, PHASE 0: CPU-only preparation. No GPU, so no lease needed.
#
#  1. archive the previous run so the delta comparison survives the new one
#  2. confirm the source really carries exp_26's landed default
#  3. rebuild and record the new module sha256, asserting it DIFFERS from the
#     pre-exp_26 binary 79599cce -- a stale build silently reproducing the old
#     numbers is the failure mode that would read as "exp_26 did nothing"
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
OLD_SHA=79599cce
PREV=$D/prev_79599cce

echo "=============== 1. archive the previous run ==============="
mkdir -p "$PREV"
if [ -d "$D/raw/ladder" ] && [ -n "$(ls "$D/raw/ladder" 2>/dev/null)" ]; then
  cp -a "$D/raw/ladder"      "$PREV/raw_ladder"      2>/dev/null
  cp -a "$D/raw/eval"        "$PREV/raw_eval"        2>/dev/null
  cp -f "$D/ladders.json"    "$PREV/ladders.json"    2>/dev/null
  cp -f "$D/logs/provenance.txt" "$PREV/provenance.txt" 2>/dev/null
  echo "archived to $PREV:"
  echo "  rank files : $(ls "$PREV/raw_ladder"/lad_s*.rank*.json 2>/dev/null | wc -l)"
  echo "  ladders.json: $(stat -c%s "$PREV/ladders.json" 2>/dev/null) bytes"
else
  echo "WARNING: no previous raw/ladder to archive"
fi

echo
echo "=============== 2. is exp_26's default really in the source? ==============="
grep -n -A3 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' "$ON/../gemm_rs_mi300x.cpp"
echo "--- the rgroup select itself ---"
grep -n -B2 -A14 'rgroup' "$ON/../gemm_rs_mi300x.cpp" | sed -n '1,60p'

echo
echo "=============== 3. sha256 BEFORE the rebuild ==============="
for f in gemm_rs_mi300x.so dhk_rt.so; do
  [ -f "$ON/harness/build/$f" ] && \
    echo "  $f $(sha256sum "$ON/harness/build/$f" | cut -c1-16) mtime $(date -Is -r "$ON/harness/build/$f")"
done

echo
echo "=============== 4. rebuild ==============="
docker exec dhk-gemmrs bash "$ON/harness/build.sh" 2>&1 | tail -18

echo
echo "=============== 5. sha256 AFTER, and the assertion ==============="
NEW=$(sha256sum "$ON/harness/build/gemm_rs_mi300x.so" | cut -c1-16)
echo "  gemm_rs_mi300x.so $NEW  mtime $(date -Is -r "$ON/harness/build/gemm_rs_mi300x.so")"
echo "  full sha256: $(sha256sum "$ON/harness/build/gemm_rs_mi300x.so" | cut -d' ' -f1)"
case "$NEW" in
  ${OLD_SHA}*)
    echo "  FATAL: module sha still starts with $OLD_SHA -- this is the PRE-exp_26"
    echo "         binary. The rebuild did not take. Do NOT measure with this."
    exit 1;;
  *)
    echo "  OK: differs from the pre-exp_26 $OLD_SHA... as required";;
esac

echo
echo "=============== 6. M2 resources/ISA at the new build ==============="
docker exec dhk-gemmrs bash "$ON/tools/m2_report.sh" 2>&1 | tail -25
