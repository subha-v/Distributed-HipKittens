#!/usr/bin/env bash
# Detach the GPU phase from this ssh session. A dropped Windows-side connection
# must not orphan a half-run campaign that is holding the node lease: setsid
# gives it its own session, and the trap inside go_gpu.sh releases the lease on
# any exit path including the timeout's SIGTERM.
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline
LOG=$E22/logs/go_gpu.log
mkdir -p "$E22/logs"
if pgrep -f "go_gpu.sh" > /dev/null; then
  echo "already running:"; pgrep -af "go_gpu.sh"; exit 0
fi
: > "$LOG"
setsid timeout 5400 env EXP22_REUSE_M7="${EXP22_REUSE_M7:-0}" \
  bash "$E22/go_gpu.sh" all >> "$LOG" 2>&1 < /dev/null &
sleep 5
echo "launched pid $! ; log=$LOG"
tail -5 "$LOG"
