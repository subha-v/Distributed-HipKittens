#!/usr/bin/env bash
# Read-only progress poll for whatever exp_20 job is detached right now.
#   poll.sh [pidfile_basename] [tail_lines]
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
PIDF=${1:-.abl.pid}
N=${2:-40}
case "$PIDF" in
  .prof.pid) LOG=$OUT/counters_run.log ;;
  .m7.pid)   LOG=$OUT/m7_run.log ;;
  *)         LOG=$OUT/reattribute_run.log ;;
esac

if [ -f "$OUT/$PIDF" ] && kill -0 "$(cat "$OUT/$PIDF")" 2>/dev/null; then
  echo "STATUS: running pid=$(cat "$OUT/$PIDF")"
else
  echo "STATUS: not running (finished or never started)"
fi
echo "KFD pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
echo "--- tail -$N $LOG ---"
tail -n "$N" "$LOG" 2>&1
