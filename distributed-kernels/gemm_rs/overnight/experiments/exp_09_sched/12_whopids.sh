#!/usr/bin/env bash
# Who is on the GPUs? Wait for them to drain (do NOT kill blindly -- HIP IPC
# mappings leak if a peer is SIGKILLed mid-flight).
set -uo pipefail
WAIT=${1:-0}
for i in $(seq 0 "$WAIT"); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  echo "[$(date +%H:%M:%S)] KFD pids: $n"
  [ "$n" = "0" ] && { echo "DRAINED"; exit 0; }
  if [ "$i" = "0" ]; then
    rocm-smi --showpids 2>&1 | sed -n '1,30p'
    echo "-- host view --"
    for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
      printf '%-8s %s\n' "$p" "$(ps -o user=,etime=,args= -p "$p" 2>/dev/null | cut -c1-140)"
      printf '         container: %s\n' "$(cat /proc/$p/cgroup 2>/dev/null | grep -o 'docker[-/][0-9a-f]\{12\}' | head -1)"
    done
  fi
  sleep 10
done
echo "STILL BUSY after $((WAIT*10))s"
exit 1
