#!/usr/bin/env bash
# Read-only: is the orphaned M9 run making progress, or is it wedged?
set -uo pipefail
A=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_26_release_pershape
H=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness

echo "===== exp_26 logs (newest first) ====="
ls -lt --time-style=+%H:%M:%S "$A"/*.log "$A"/*.txt "$A"/*.json 2>/dev/null | head -10

echo
echo "===== tail of the newest exp_26 log ====="
f=$(ls -t "$A"/*.log 2>/dev/null | head -1)
if [ -n "${f:-}" ]; then
  echo "### $f  (mtime $(stat -c %y "$f"))"
  tail -30 "$f" 2>&1
  echo "-- seconds since last write: $(( $(date +%s) - $(stat -c %Y "$f") )) --"
else
  echo "(no exp_26 log)"
fi

echo
echo "===== any m9 output under harness ====="
ls -lt --time-style=+%H:%M:%S "$H"/m9_*.json "$H"/m9_*.log 2>/dev/null | head -5

echo
echo "===== process state / progress indicators ====="
for p in 3001610; do
  echo "--- pid $p ---"
  cat /proc/$p/status 2>/dev/null | grep -E 'State|Threads|VmRSS'
  echo "  wchan: $(cat /proc/$p/wchan 2>/dev/null)"
  echo "  fds:   $(ls /proc/$p/fd 2>/dev/null | wc -l)"
  echo "  utime/stime ticks: $(awk '{print $14, $15}' /proc/$p/stat 2>/dev/null)"
  sleep 5
  echo "  after 5s ->      $(awk '{print $14, $15}' /proc/$p/stat 2>/dev/null)  (if unchanged, it is not burning CPU)"
done

echo
echo "===== GPU occupancy sample over 10s ====="
for i in 1 2 3; do
  rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{printf "  sample: pid %s cu_occupancy %s\n", $1, $6}'
  sleep 5
done
echo done
