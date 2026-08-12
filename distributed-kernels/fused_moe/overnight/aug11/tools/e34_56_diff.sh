#!/usr/bin/env bash
# exp_34 BLOCKING diagnostic part 2: g=353 means "throttle on + exp_24 depth 4"
# (low byte 0x61 | high byte 0x01) and g=65 means unthrottled. rev 26 measured
# g=353 -> 6,488 and g=65 -> 7,111; the pin measures g=353 -> 7,222. So the
# question is whether rev 27/28 changed the code that consumes the throttle /
# depth bits. Diff ONLY the throttle-relevant hunks, f113d73f..291dfa08.
set -uo pipefail
cd "$HOME/Distributed-HipKittens"
F=distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
A=distributed-kernels/fused_moe/moe_mps_adapter.cuh
echo "===commits between rev26 and the pin==="
git log --oneline f113d73f..291dfa08 -- "$F" "$A" | cut -c1-108
echo
echo "===diffstat==="
git diff --stat f113d73f 291dfa08 -- "$F" "$A"
echo
echo "===throttle / depth / group_slices hunks in the diff==="
git diff -U6 f113d73f 291dfa08 -- "$F" "$A" \
  | grep -n -E "^(@@|[-+].*(throttle|depth|0x20|0x10u|0x0F|group_slices|inflight|in_flight|outstanding))" \
  | cut -c1-170 | head -80
echo
echo "===where the throttle is consumed at the PIN==="
git grep -n -E "throttle|in_flight|inflight|depth_sel|kThrottle" 291dfa08 -- "$F" | cut -c1-150 | head -40
echo "===DONE==="
