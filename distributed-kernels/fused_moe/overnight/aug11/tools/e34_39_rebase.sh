#!/usr/bin/env bash
# exp_34: does origin's new HEAD touch either owned file since my base ca5b683f?
set -uo pipefail
cd $HOME/Distributed-HipKittens
echo "### commits since ca5b683f ###"
git log --oneline ca5b683f..HEAD
echo
echo "### did any of them touch the two owned files? ###"
git log --oneline ca5b683f..HEAD -- \
  distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
  distributed-kernels/fused_moe/moe_mps_adapter.cuh
echo "(empty above = my diff rebases cleanly onto HEAD)"
echo
echo "### diffstat of the range on those files ###"
git diff --stat ca5b683f..HEAD -- distributed-kernels/fused_moe/ | tail -20
echo "===DONE==="
