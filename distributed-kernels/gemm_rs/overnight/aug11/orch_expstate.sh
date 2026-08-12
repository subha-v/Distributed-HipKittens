#!/usr/bin/env bash
# Read-only: per-experiment progress across the aug11 queue.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
A=$ON/aug11

echo "===== lease ====="
bash "$ON/tools/gpu_lease.sh" status
tail -8 "$A/gpu_lease.log" 2>/dev/null

echo
echo "===== exp_23 waterfall: draws + results ====="
ls -la --time-style=+%H:%M "$A"/exp_23_waterfall/draw_*.json "$A"/exp_23_waterfall/waterfall.json 2>&1 | tail -12
echo "-- newest sweep log tail --"
for f in $(ls -t "$A"/exp_23_waterfall/*.log 2>/dev/null | head -2); do
  echo "### $f"
  tail -25 "$f" 2>&1
done

echo
echo "===== exp_21 saturation ====="
ls -la --time-style=+%H:%M "$A"/exp_21_saturation/saturation*.json "$A"/exp_21_saturation/*.log 2>&1 | tail -8

echo
echo "===== exp_22 timeline ====="
ls -la --time-style=+%H:%M "$A"/exp_22_timeline/ 2>&1 | tail -12

echo
echo "===== exp_24 ladders ====="
ls -la --time-style=+%H:%M "$A"/exp_24_ladders/*.json "$A"/exp_24_ladders/raw 2>&1 | tail -10

echo
echo "===== running processes ====="
ps -eo pid,etime,stat,cmd 2>/dev/null | grep -E 'sweep|ladder|saturation|trace|hipcc|python3 -u' | grep -v grep | head -20 || echo "(none)"
echo done
