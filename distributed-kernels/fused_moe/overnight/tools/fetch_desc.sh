#!/usr/bin/env bash
set -uo pipefail
LOG="$HOME/exp01_descdump.log"
OUT="$HOME/k0-mok-mps-descdump"

echo "########## DESC DUMP FILES ##########"
for f in "$OUT"/run1/mps_desc_rank*.txt; do
  echo "----- $f -----"
  cat "$f"
done

echo "########## PROGRESS rank0 ##########"
cat "$OUT"/run1/progress_rank0.log

echo "########## LOG: error context ##########"
grep -n -i -E 'error|fault|abort|assert|Traceback|RuntimeError|nil|exitcode|slots|shmem|heap' "$LOG" | head -60
