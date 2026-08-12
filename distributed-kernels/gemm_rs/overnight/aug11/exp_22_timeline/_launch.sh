#!/usr/bin/env bash
# Detach the GPU phase from this ssh session. A dropped Windows-side connection
# must not orphan a half-run campaign holding the node lease: setsid gives it
# its own session, and the trap inside the runner releases the lease on any exit
# path including the timeout's SIGTERM.
#
# The outer timeout has to cover the QUEUE WAIT as well as the run. Sizing it to
# the run alone is how the last attempt died: it expired waiting behind exp_24.
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
SCRIPT=${EXP22_SCRIPT:-go_armb.sh}
LOG=$E22/logs/${SCRIPT%.sh}.log
mkdir -p "$E22/logs"
if pgrep -f "$SCRIPT" > /dev/null; then
  echo "already running:"; pgrep -af "$SCRIPT"; exit 0
fi
: > "$LOG"
setsid timeout "${EXP22_TIMEOUT:-21600}" bash "$E22/$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
sleep 6
echo "launched pid $! ; log=$LOG"
tail -12 "$LOG"
