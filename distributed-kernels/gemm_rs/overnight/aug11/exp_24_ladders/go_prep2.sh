#!/usr/bin/env bash
# exp_24 re-measure, PHASE 0b: repair the CRLF damage, then rebuild for real.
#
# go_prep.sh's build died mid-script because harness/build.sh and
# tools/m2_report.sh arrived with Windows CRLF -- the same defect that cost two
# launches earlier tonight (LESSONS: tools/gpu_lease.sh CRLF). Only the FIRST
# module built before the script hit a \r; the "sha differs" line it printed is
# therefore not a verdict I am willing to measure against. Strip, syntax-check,
# rebuild all three modules, and re-assert.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OLD_SHA=79599cce

echo "=============== which files carry CR? ==============="
CRFILES=$(grep -rlU $'\r' "$ON/harness" "$ON/tools" "$ON/aug11/exp_24_ladders" \
            --include='*.sh' --include='*.py' 2>/dev/null | sort)
if [ -z "$CRFILES" ]; then echo "  (none)"; else echo "$CRFILES" | sed 's/^/  /'; fi

echo
echo "=============== strip CR in place (my own tree) ==============="
for f in $CRFILES; do
  tr -d '\r' < "$f" > "$f.nocr" && cat "$f.nocr" > "$f" && rm -f "$f.nocr"
  case "$f" in
    *.sh) bash -n "$f" && echo "  OK  bash -n  $(basename "$f")" || { echo "  FATAL syntax $f"; exit 1; };;
    *.py) python3 -m py_compile "$f" && echo "  OK  compile  $(basename "$f")" || { echo "  FATAL syntax $f"; exit 1; };;
  esac
done
echo "residual CR files: $(grep -rlU $'\r' "$ON/harness" "$ON/tools" "$ON/aug11/exp_24_ladders" --include='*.sh' --include='*.py' 2>/dev/null | wc -l)"

echo
echo "=============== force a from-scratch rebuild ==============="
# Remove the objects so a stale artifact cannot survive as a false pass.
for f in gemm_rs_mi300x.so gemm_rs_mi300x_control.so dhk_rt.so; do
  rm -f "$ON/harness/build/$f"
done
ls -la "$ON/harness/build/" | sed 's/^/  /'
docker exec dhk-gemmrs bash "$ON/harness/build.sh" > "$ON/aug11/exp_24_ladders/logs/rebuild.log" 2>&1
rc=$?
echo "build rc=$rc"
grep -E '^(OK|FAIL|#{5,})' "$ON/aug11/exp_24_ladders/logs/rebuild.log" | sed 's/^/  /'
if [ $rc -ne 0 ]; then
  echo "FATAL: build failed; last 30 lines:"; tail -30 "$ON/aug11/exp_24_ladders/logs/rebuild.log"; exit 1
fi

echo
echo "=============== module identity under test ==============="
for f in gemm_rs_mi300x.so gemm_rs_mi300x_control.so dhk_rt.so; do
  p="$ON/harness/build/$f"
  if [ -f "$p" ]; then
    echo "  $f  $(sha256sum "$p" | cut -d' ' -f1)  $(stat -c%s "$p") B  $(date -Is -r "$p")"
  else
    echo "  FATAL missing $f"; exit 1
  fi
done
NEW=$(sha256sum "$ON/harness/build/gemm_rs_mi300x.so" | cut -d' ' -f1)
case "$NEW" in
  ${OLD_SHA}*) echo "  FATAL: still the PRE-exp_26 binary $OLD_SHA -- refusing to measure"; exit 1;;
  *)           echo "  ASSERT OK: ${NEW:0:16} != ${OLD_SHA}... (pre-exp_26)";;
esac
echo "$NEW" > "$ON/aug11/exp_24_ladders/logs/module_sha_under_test.txt"

echo
echo "=============== M2 resources / ISA ==============="
docker exec dhk-gemmrs bash "$ON/tools/m2_report.sh" 2>&1 | tail -30
