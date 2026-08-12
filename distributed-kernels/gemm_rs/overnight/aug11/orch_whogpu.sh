#!/usr/bin/env bash
# Read-only: identify every process currently holding a GPU, leased or not.
set -uo pipefail
echo "===== rocm-smi --showpids ====="
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'

echo
echo "===== full cmdline for each KFD pid ====="
for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
  echo "--- pid $p ---"
  ps -o pid,ppid,etime,user,stat --no-headers -p "$p" 2>&1
  tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-400; echo
  echo "  cwd: $(readlink -f /proc/$p/cwd 2>/dev/null)"
  # parent chain, to attribute it to an experiment
  pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  for i in 1 2 3; do
    [ -z "$pp" ] || [ "$pp" = "1" ] || [ "$pp" = "0" ] && break
    echo "  parent $pp: $(tr '\0' ' ' < /proc/$pp/cmdline 2>/dev/null | cut -c1-200)"
    pp=$(ps -o ppid= -p "$pp" 2>/dev/null | tr -d ' ')
  done
done

echo
echo "===== any python/hipcc under the aug11 tree ====="
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep -E 'aug11|dhk' | grep -v grep | cut -c1-220 | head -20 || echo "(none)"
echo done
