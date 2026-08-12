#!/usr/bin/env bash
# exp_34: print the throttle_depth_sel hunk (and its neighbours) in full,
# f113d73f -> 291dfa08, plus the function bodies at both revisions.
set -uo pipefail
cd "$HOME/Distributed-HipKittens"
A=distributed-kernels/fused_moe/moe_mps_adapter.cuh
echo "############ DIFF around throttle_depth_sel ############"
git diff -U22 f113d73f 291dfa08 -- "$A" | sed -n '/throttle_depth_sel/,/^@@/p' | head -70
echo
for REV in f113d73f 291dfa08; do
  echo "############ $REV : throttle accessors as compiled ############"
  git show "$REV:$A" | grep -n -A16 -E "throttle_depth_sel|throttle_enabled|remote_accum" | \
      sed -n '1,90p'
  echo
done
echo "===DONE==="
