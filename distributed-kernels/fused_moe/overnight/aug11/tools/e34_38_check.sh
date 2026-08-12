#!/usr/bin/env bash
# exp_34: prove the pinned node checkout was never touched.
set -uo pipefail
echo "### pinned checkout (belongs to the running GPU campaign) ###"
cd $HOME/Distributed-HipKittens 2>/dev/null && {
  git log --oneline -1
  echo "--- porcelain (must be empty or unrelated to fused_moe) ---"
  git status --porcelain | head -20
  echo "--- mtimes of the two owned files there ---"
  ls -l --time-style=full-iso distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
                              distributed-kernels/fused_moe/moe_mps_adapter.cuh
  echo "--- does it contain mode 14? (must be NO) ---"
  grep -c "kModeCoarseReady" distributed-kernels/fused_moe/moe_mps_adapter.cuh
}
echo "===DONE==="
