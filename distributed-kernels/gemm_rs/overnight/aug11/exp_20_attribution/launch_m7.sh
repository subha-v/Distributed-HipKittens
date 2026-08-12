#!/usr/bin/env bash
# Detach the M7 gate run. Expects run_m7.sh staged on the node.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
sed -i 's/\r$//' "$OUT"/*.sh "$OUT"/*.py 2>/dev/null
if [ -f "$OUT/.m7.pid" ] && kill -0 "$(cat "$OUT/.m7.pid")" 2>/dev/null; then
  echo "ALREADY RUNNING pid=$(cat "$OUT/.m7.pid")"; exit 0
fi
echo "KFD pids before launch: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
rm -f "$OUT/m7_run.log"
setsid nohup bash "$OUT/run_m7.sh" 3 50 > "$OUT/m7_run.log" 2>&1 &
echo $! > "$OUT/.m7.pid"
sleep 6
echo "launched pid=$(cat "$OUT/.m7.pid")"
sed -n '1,10p' "$OUT/m7_run.log"
