#!/usr/bin/env bash
# exp_34: the throttle for mode 12 is implemented by k0p6_defer +
# k0p6_mps_task_flush_defer + k0p6_mps_task_done. My mode-14 commit edited all
# three. Print those hunks in full so a mode-12 behaviour change is visible.
set -uo pipefail
cd "$HOME/Distributed-HipKittens"
F=distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip
git diff -U10 f113d73f 291dfa08 -- "$F" | awk '
  /^@@/ { p = ($0 ~ /task_done|k0p6_defer|task_flush_defer/) }
  p { print }
' | head -160
echo "===DONE==="
