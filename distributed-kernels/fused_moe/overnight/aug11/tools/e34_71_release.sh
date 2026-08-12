#!/usr/bin/env bash
# exp_34 wrap-up: dump the final JSON to stdout for retrieval, confirm the pin is
# intact, and confirm the node is idle so the lease can be released.
set -uo pipefail
echo "===PIN==="
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
git -C "$HOME/Distributed-HipKittens" status --porcelain | head -5
echo "(empty porcelain = clean)"
echo "===IDLE CHECK==="
pgrep -af "screen.sh|torchrun|run_campaign|mok_synthetic" | head -10 || echo "(no jobs)"
rocm-smi --showpids 2>/dev/null | sed -n '1,20p'
echo "===JSON BEGIN==="
cat "$HOME/e34/mode14_arms.json"
echo "===JSON END==="
