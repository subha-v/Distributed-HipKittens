#!/usr/bin/env bash
# t17: one-line batch launcher. Edit tag + expected mask per batch.
# First: relabel the batch that was tagged e26_m5 but provably ran mask 1
# (.text 9d9b6539... on all three of its jit dirs) because my mask-5 push was
# rejected by a concurrent push and the launch went ahead on the old tree.
S=$HOME/overnight-scratch
if [ -f "$S/screen_e26_m5.csv" ] && [ ! -f "$S/screen_e26_m1b.csv" ]; then
  mv "$S/screen_e26_m5.csv" "$S/screen_e26_m1b.csv"
  echo "relabelled screen_e26_m5.csv -> screen_e26_m1b.csv (arm was mask 1, verified by .text)"
  echo "NOTE: its per-run logs and output dirs keep the misleading e26_m5_* names."
fi
bash /home/subvadla/tools/ladder.sh e26_m5 5 5
exit 0
