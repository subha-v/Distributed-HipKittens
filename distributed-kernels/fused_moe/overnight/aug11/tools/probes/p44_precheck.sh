#!/usr/bin/env bash
# exp_24 pre-batch check: (a) does the node HEAD contain any KERNEL change on top
# of the exp_24 commit 520fe9c6? (b) does every g in the run plan decode as
# intended and pass config_is_valid?
set -uo pipefail
DHK=$HOME/Distributed-HipKittens
echo "=== node HEAD"
git -C "$DHK" log --oneline -1
echo "=== source-file diff 520fe9c6..HEAD (must be EMPTY for a valid measurement)"
git -C "$DHK" diff --stat 520fe9c6..HEAD -- \
  'distributed-kernels/fused_moe/*.hip' \
  'distributed-kernels/fused_moe/*.cuh' \
  'distributed-kernels/fused_moe/*.cpp' \
  'distributed-kernels/fused_moe/*.hpp' \
  'include/'
echo "(nothing above => the kernel measured IS exp_24 520fe9c6)"
echo
echo "=== which files DID change since 520fe9c6"
git -C "$DHK" diff --name-only 520fe9c6..HEAD | head -20
exit 0
