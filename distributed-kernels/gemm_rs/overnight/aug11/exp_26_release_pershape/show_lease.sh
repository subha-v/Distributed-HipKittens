#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
sed -n '55,200p' "$ON/tools/gpu_lease.sh"
echo "===== state ====="
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head
echo "===== stuck proc ====="
for p in $(pgrep -f 'm9_stale_slot'); do
  echo "pid $p state=$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null) wchan=$(cat /proc/$p/wchan 2>/dev/null)"
done
pgrep -f 'm9_stale_slot' >/dev/null || echo "  (gone)"
