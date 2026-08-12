#!/usr/bin/env bash
# Is the unleased job on the GPUs making progress, or is it wedged?
#
# Two samples 75 s apart of CPU time, syscall/wchan state, and its log's size, for
# every KFD pid. A job with zero CU occupancy AND flat CPU time AND a log that has
# not grown is not computing; it is waiting on something that is not coming. This
# only reads /proc and the log -- it signals nothing.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
HARNESS=$ON/harness

pids=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}')
[ -z "$pids" ] && { echo "no KFD pids"; exit 0; }

sample() {
  for p in $pids; do
    [ -d "/proc/$p" ] || { echo "  pid $p GONE"; continue; }
    ut=$(awk '{print $14" "$15}' "/proc/$p/stat" 2>/dev/null)
    st=$(awk '/^State/{print $2$3}' "/proc/$p/status" 2>/dev/null)
    th=$(awk '/^Threads/{print $2}' "/proc/$p/status" 2>/dev/null)
    wc=$(cat "/proc/$p/wchan" 2>/dev/null)
    echo "  pid $p state=$st threads=$th utime/stime=$ut wchan=${wc:-none}"
  done
  echo "  -- rocm-smi CU occupancy --"
  rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]+' | sed 's/^/    /'
  echo "  -- newest harness logs --"
  ls -la --time-style=+%H:%M:%S "$HARNESS"/*.log "$HARNESS"/logs/*.log 2>/dev/null \
    | sort -k6 | tail -5 | sed 's/^/    /'
  ls -la --time-style=+%H:%M:%S "$ON"/logs/*m9* "$ON"/aug11/**/m9* 2>/dev/null | sed 's/^/    /'
}

echo "########## sample 1 ($(date -u +%H:%M:%SZ)) ##########"
sample
echo
echo "sleeping 75s"
sleep 75
echo
echo "########## sample 2 ($(date -u +%H:%M:%SZ)) ##########"
sample

echo
echo "########## m9_stale_slot.py: what does it wait on ##########"
grep -nE 'while|sleep|timeout|wait|input|barrier|spin' "$HARNESS/m9_stale_slot.py" 2>/dev/null \
  | head -25 || echo "  (source not found at $HARNESS/m9_stale_slot.py)"
