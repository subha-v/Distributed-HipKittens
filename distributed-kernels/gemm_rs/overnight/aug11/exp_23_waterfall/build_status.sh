#!/usr/bin/env bash
# Poll the detached exp_23 build. Prints whether it is still running, the
# artifacts it has produced so far, and the tail of the log.
#   nsh.ps1 -Script ...\exp_23_waterfall\build_status.sh -ArgLine "80"
set -uo pipefail

EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
LINES=${1:-60}

if pgrep -f 'exp_23_waterfall/build_rungs.sh' >/dev/null 2>&1 || \
   pgrep -f 'exp_23_waterfall/fingerprint.py' >/dev/null 2>&1; then
  echo "STATE: running"
else
  echo "STATE: not running"
fi

echo
echo "-- artifacts --"
ls -la "$EXP/build" 2>/dev/null | grep -E '\.so$|\.fallback$' || echo "  (no .so yet)"
ls -la "$EXP/build/isa" 2>/dev/null | grep -E '\.s$' || echo "  (no .s yet)"
[ -f "$EXP/fingerprints.json" ] && \
  echo "  fingerprints.json $(stat -c%s "$EXP/fingerprints.json") bytes" || \
  echo "  fingerprints.json: absent"

echo
echo "-- build.log (last $LINES lines) --"
tail -n "$LINES" "$EXP/build.log" 2>/dev/null || echo "  (no log)"
