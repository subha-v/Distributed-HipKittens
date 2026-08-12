#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
python3 "$D/inspect_quick.py" 2>&1
echo
echo "=== raw quick artifacts ==="
ls -l "$D/raw/ladder_quick/" 2>&1 | head -20
echo
echo "=== any SHIM_WAS_CALLED / fault / traceback in the quick run stderrs? ==="
grep -lE 'SHIM_WAS_CALLED|Memory access fault|Traceback' "$D/raw/ladder_quick/"*.stderr 2>/dev/null || echo "none -- clean"
echo
echo "=== is the node copy a git worktree? (provenance printed empty) ==="
git -C /home/subvadla/dhk rev-parse --is-inside-work-tree 2>&1 | head -2
