#!/usr/bin/env bash
# exp_30 stage 1: read the grid-barrier primitive + the ladder launcher.
set -uo pipefail
F=/home/subvadla/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_sync.hpp
echo "=== hkp_sync.hpp lines 30-70 (fail_closed) ==="
sed -n '30,70p' "$F"
echo
echo "=== hkp_sync.hpp lines 200,260 (local_grid_epoch + grid_barrier) ==="
sed -n '200,262p' "$F"
echo
echo "=== ladder.sh ==="
cat /home/subvadla/tools/ladder.sh
echo
echo "=== e27_fp.sh (text fingerprint tool) ==="
cat /home/subvadla/tools/e27_fp.sh
