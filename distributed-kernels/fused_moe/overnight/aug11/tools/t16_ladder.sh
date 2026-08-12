#!/usr/bin/env bash
# t16 (pushed to ~/tools/ladder.sh): run N repeats of the ratchet config with
# timestamps on, as one screen batch. Usage: ladder.sh TAG NREPS
# One batch == one build == one mask, because the phase-1 hint mask is a
# compile-time #define; the node checkout is synced from origin once per batch.
set -uo pipefail
TAG="${1:?usage: ladder.sh TAG NREPS}"
N="${2:-5}"
CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1"
LOG="$HOME/overnight-scratch/${TAG}.driver.log"

echo "== pre-flight =="
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
if pgrep -f 'tools/screen.sh' >/dev/null 2>&1; then
  echo "REFUSING: a screen batch is already running"; exit 9
fi

: > /tmp/${TAG}.cfg
for i in $(seq 1 "$N"); do echo "$CFG" >> /tmp/${TAG}.cfg; done
echo "$TAG" > /tmp/screen_tag
echo "== $N repeats of: $CFG =="

rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1 \
  SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100 SCREEN_SYNC=1 \
  bash "$HOME/tools/screen.sh" "$TAG" /tmp/${TAG}.cfg \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 25
head -25 "$LOG"
exit 0
