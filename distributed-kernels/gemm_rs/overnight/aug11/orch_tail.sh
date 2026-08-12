#!/usr/bin/env bash
# Read-only: what are exp_22 and exp_24 actually doing right now?
set -uo pipefail
A=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11

for e in exp_22_timeline exp_24_ladders; do
  echo "================ $e ================"
  echo "--- newest 8 files ---"
  ls -lt --time-style=+%H:%M:%S "$A/$e" 2>/dev/null | head -9
  echo "--- newest log/txt tail ---"
  f=$(ls -t "$A/$e"/*.log "$A/$e"/*.txt 2>/dev/null | head -1)
  if [ -n "${f:-}" ]; then
    echo "### $(basename "$f")  last write $(( $(date +%s) - $(stat -c %Y "$f") ))s ago"
    tail -22 "$f" 2>&1
  else
    echo "(no log)"
  fi
  echo "--- json present? ---"
  ls -la --time-style=+%H:%M "$A/$e"/*.json "$A/$e"/*.csv 2>/dev/null | tail -6
  echo
done

echo "================ any live process, broad match ================"
ps -eo pid,etime,stat,cmd --no-headers 2>/dev/null \
  | grep -E 'python3|torchrun|rocprof|hipcc|bash .*aug11' | grep -v grep | cut -c1-200 | head -25 || echo "(none)"

echo
echo "================ lease ================"
bash /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/gpu_lease.sh status
echo done
