#!/usr/bin/env bash
# exp_34: remove a stray scp target created by an early mistyped command, and
# confirm the pinned checkout was never touched. READ/CLEAN ONLY. NO GPU WORK.
set -uo pipefail
rm -rf "$HOME/e34/incoming_kernel_adapter_tmp"
rm -rf "$HOME/\$HOME" 2>/dev/null || true
echo "=== ~/e34 top level ==="
ls -1 "$HOME/e34"
echo "=== the PINNED checkout must be untouched ==="
cd "$HOME/Distributed-HipKittens" && git log --oneline -1 && git status --porcelain | head
echo "=== our scratch clone ==="
cd "$HOME/e34/DHK" && git log --oneline -1 && git status --porcelain
echo "===DONE==="
