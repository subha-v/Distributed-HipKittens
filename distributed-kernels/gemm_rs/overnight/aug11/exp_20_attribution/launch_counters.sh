#!/usr/bin/env bash
# aug11/exp_20: launch the counter pass DETACHED, so an ssh hiccup cannot kill
# a ~20 minute GPU job. Poll with `snap.sh <tag>` / `poll.sh .prof.pid`.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution

if [ -f "$OUT/.prof.pid" ] && kill -0 "$(cat "$OUT/.prof.pid")" 2>/dev/null; then
  echo "ALREADY RUNNING pid=$(cat "$OUT/.prof.pid")"; exit 0
fi
if [ -f "$OUT/.abl.pid" ] && kill -0 "$(cat "$OUT/.abl.pid")" 2>/dev/null; then
  echo "REFUSING: the attribution run is still live; one GPU job at a time."
  exit 1
fi

echo "KFD pids before launch: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
rm -f "$OUT/counters_run.log"
setsid nohup timeout 5400 bash "$OUT/run_counters.sh" 2 4 \
  > "$OUT/counters_run.log" 2>&1 &
echo $! > "$OUT/.prof.pid"
sleep 8
echo "launched pid=$(cat "$OUT/.prof.pid")"
tail -n 15 "$OUT/counters_run.log"
