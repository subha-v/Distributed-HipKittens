#!/usr/bin/env bash
# Detached launcher for the rematch measurement. Runs under setsid+nohup so an
# ssh drop does not kill an hour of GPU work; poll with rematch_status.sh.
#
# Order of business:
#   1. pin clocks (idle sclk is ~120 MHz here; unpinned short runs are not
#      repeatable)
#   2. move the pre-fix vs_logs aside -- vs_report.py globs the directory, so a
#      single stale JSON would silently poison the aggregate
#   3. REBUILD gemm_rs_mi300x / dhk_rt from the committed source. The .so on the
#      node was built 19:14 CDT and the has_bias fix was committed 20:19 CDT;
#      that ordering is consistent with build->gate->commit, but it is not
#      *proof* the fix is in the binary, and a push has since reset the source
#      mtimes so the staleness check is no longer decisive. Rebuilding from the
#      committed source removes the doubt for ~4 minutes of hipcc.
#   4. run the sweep
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1
LOG=$E/rematch_run.log
SHAPES=${1:-5,3,0,1,2,4}
ITERS=${2:-12}
REPS=${3:-2}

run_all() {
  echo "===== launcher start $(date -Is) ====="

  echo "--- pin clocks ---"
  bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -3
  rocm-smi --showperflevel 2>&1 | grep -iE 'GPU\[0\]|GPU\[7\]' | head -4

  echo "--- archive pre-fix vs_logs ---"
  if compgen -G "$E/vs_logs/*" > /dev/null; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$E/vs_logs_prefix_$STAMP"
    mv "$E/vs_logs"/* "$E/vs_logs_prefix_$STAMP"/ 2>/dev/null
    echo "moved $(ls "$E/vs_logs_prefix_$STAMP" | wc -l) files to vs_logs_prefix_$STAMP"
  else
    echo "vs_logs already empty"
  fi

  echo "--- rebuild from committed source ---"
  bash "$ON/harness/build.sh" 2>&1 | grep -E '^OK|^FAIL|^####|ALL MODULES|BUILD FAILURES|entry points'
  RC=${PIPESTATUS[0]}
  echo "build rc=$RC"
  if [ "$RC" != "0" ]; then
    echo "BUILD FAILED -- refusing to measure a binary of unknown provenance"
    return 1
  fi
  ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so" \
    "$ON/harness/build/dhk_rt.so"

  echo "--- sweep ---"
  bash "$E/rematch_sweep.sh" "$SHAPES" "$ITERS" "$REPS" 12700 1
  echo "===== launcher done $(date -Is) ====="
}

run_all > "$LOG" 2>&1
