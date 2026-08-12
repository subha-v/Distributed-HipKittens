#!/usr/bin/env bash
# Second reap attempt + diagnosis of why the first did not take, and a read of
# the current lease state (which this experiment failed to acquire and must from
# here on).
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "########## lease state ##########"
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -20
echo "-- lease log tail --"
tail -8 "$ON/aug11/gpu_lease.log" 2>/dev/null || echo "  (none)"

echo
echo "########## stuck process state ##########"
for p in $(pgrep -f 'm9_stale_slot'); do
  echo "-- pid $p --"
  awk '/^State:|^Name:|^SigIgn:|^SigCgt:/{print "   "$0}' /proc/$p/status 2>/dev/null
  echo "   wchan: $(cat /proc/$p/wchan 2>/dev/null)"
done

echo
echo "########## dmesg tail (driver's view of the fault) ##########"
dmesg 2>/dev/null | tail -20 || echo "  (dmesg not readable)"

echo
echo "########## TERM again, then wait ##########"
pids=$(pgrep -f 'm9_stale_slot' | tr '\n' ' ')
if [ -n "$pids" ]; then
  kill -s TERM $pids 2>/dev/null
  for i in $(seq 1 12); do
    sleep 15
    left=$(pgrep -f 'm9_stale_slot' | tr '\n' ' ')
    [ -z "$left" ] && { echo "  gone after $((i*15))s"; break; }
    [ $((i % 4)) = 0 ] && echo "  still there at $((i*15))s: $left"
  done
fi

echo
echo "########## after ##########"
pgrep -af 'm9_stale_slot' || echo "  m9 is gone"
echo "KFD pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
