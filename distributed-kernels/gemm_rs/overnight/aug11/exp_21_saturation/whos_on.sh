#!/usr/bin/env bash
# Identify every KFD process and its command line. Read-only; touches no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash "$ON/tools/gpu_lease.sh" status
echo
for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
  echo "-- pid $p (elapsed $(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')s) --"
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-400; echo
  echo "   cwd: $(readlink /proc/$p/cwd 2>/dev/null)"
  ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  while [ -n "${ppid:-}" ] && [ "$ppid" != "1" ] && [ "$ppid" != "0" ]; do
    echo "   parent $ppid: $(tr '\0' ' ' < /proc/$ppid/cmdline 2>/dev/null | cut -c1-200)"
    ppid=$(ps -o ppid= -p "$ppid" 2>/dev/null | tr -d ' ')
  done
done
echo
echo "-- recent lease log --"
tail -12 "$ON/aug11/gpu_lease.log" 2>/dev/null
