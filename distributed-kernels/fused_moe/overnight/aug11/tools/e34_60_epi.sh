#!/usr/bin/env bash
# exp_34: the regression is in M7 (3,472 at the pin vs ~2,660 at rev 26) with
# planM6 unchanged -- i.e. in the producer epilogue, which is exactly where the
# exp_24 throttle's vmcnt bound is applied and exactly where mode 14's M7.5
# rendezvous was folded in. Print the epilogue hunks and how the throttle is
# selected at both revisions.
set -uo pipefail
cd "$HOME/Distributed-HipKittens"
F=distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
echo "############ epilogue hunks (1577..2100) ############"
git diff -U8 f113d73f 291dfa08 -- "$F" | awk '
  /^@@/ { n = $0; sub(/^@@ -[0-9]+,[0-9]+ \+/, "", n); sub(/,.*/, "", n); ln = n + 0;
          p = (ln > 1550) }
  p { print }' | head -190
echo
echo "############ how the throttle depth reaches the epilogue, both revs ############"
for REV in f113d73f 291dfa08; do
  echo "---- $REV"
  git grep -n -E "throttle_enabled|throttle_depth_sel|vmcnt\(|push_throttled|K0P6_MPS_THROTTLE" "$REV" -- "$F" \
    | cut -c1-140 | head -30
done
echo "===DONE==="
