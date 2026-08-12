#!/usr/bin/env bash
# campaign3.sh acquired the lease and then died at line 36. If the trap never
# ran, exp_26 is holding the node against every other queued experiment. Release
# it immediately, then show what actually is on line 36.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
tr -d '\r' < "$ON/tools/gpu_lease.sh" > /tmp/gpu_lease_exp26.sh

echo "===== lease before ====="
bash /tmp/gpu_lease_exp26.sh status 2>&1 | head -4

owner=$(cat "$ON/aug11/.gpu_lease/owner" 2>/dev/null || echo none)
if [ "$owner" = "exp_26" ]; then
  echo "===== releasing (ours, orphaned by the crash) ====="
  bash /tmp/gpu_lease_exp26.sh release exp_26 2>&1 | head -2
fi

echo "===== lease after ====="
bash /tmp/gpu_lease_exp26.sh status 2>&1 | head -4

echo
echo "===== campaign3.sh lines 30-42, with hidden characters ====="
sed -n '30,42p' "$D/campaign3.sh" | cat -A | sed 's/\$$/<LF>/'
