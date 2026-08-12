#!/usr/bin/env bash
# Launch the full gate ladder DETACHED (setsid + timeout), so an ssh drop cannot
# kill an 8-GPU job mid-flight and leave IPC mappings behind.
#
# The ladder writes its own logs under experiments/exp_26_release_pershape/logs
# (that path is baked into tools/gate_ladder.sh); this wrapper keeps the console
# transcript next to the experiment.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
mkdir -p "$D/logs" "$ON/experiments/exp_26_release_pershape/logs"
LOG=$D/logs/ladder.log

if pgrep -f 'gate_ladder.sh exp_26' >/dev/null 2>&1; then
  echo "REFUSED: a gate ladder for exp_26 is already running"; exit 1
fi

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids before launch: $n"
[ "$n" = "0" ] || { echo "REFUSED: node dirty"; exit 1; }

: > "$LOG"
setsid nohup timeout 16200 bash "$ON/tools/gate_ladder.sh" \
  exp_26_release_pershape >> "$LOG" 2>&1 &
echo "launched pid $! ; log $LOG"
sleep 5
tail -5 "$LOG"
