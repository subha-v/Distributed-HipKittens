#!/usr/bin/env bash
# Read-only: why is the lease-holder not using the GPU?
# Walk the holder's process tree and show what each descendant is blocked on.
set -uo pipefail

show() {
  local p=$1 depth=$2 pad
  pad=$(printf '%*s' "$depth" '')
  [ -d "/proc/$p" ] || return 0
  echo "${pad}pid $p [$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null)] wchan=$(cat /proc/$p/wchan 2>/dev/null) :: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-150)"
  for c in $(cat /proc/$p/task/*/children 2>/dev/null | tr ' ' '\n' | sort -u); do
    [ -n "$c" ] && show "$c" $((depth+2))
  done
}

for root in 3498810 3519035 3653905; do
  echo "===== tree from $root ====="
  show "$root" 0
  echo
done

echo "===== exp_24 runner: recent output, if it redirects anywhere ====="
for p in 3498961 3729783; do
  [ -d "/proc/$p" ] || continue
  echo "--- pid $p open files ---"
  ls -l /proc/$p/fd 2>/dev/null | awk '{print "   ", $9, $10, $11}' | head -8
done

echo
echo "===== newest files anywhere under aug11 (who is actually writing?) ====="
find /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11 -type f -newermt '-6 minutes' -printf '%TH:%TM:%TS  %p\n' 2>/dev/null | sort | tail -15
echo "  (nothing = no experiment has written a file in 6 minutes)"
echo done
