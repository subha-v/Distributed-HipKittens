#!/usr/bin/env bash
# exp_34: my ssh dropped mid-batch; setsid means the job survived. Is it still
# running, and how far did it get? Never start a second GPU job over the top.
set -uo pipefail
date -u
echo "===screen.sh / torchrun processes==="
pgrep -af "screen.sh|torchrun|run_campaign|mok_synthetic" | head -20 || echo "(none)"
echo
echo "===GPU pids (the check that works on this node)==="
rocm-smi --showpids 2>/dev/null | head -24
echo
echo "===e34d csv rows so far==="
grep -h "^SCREEN " "$HOME/overnight-scratch/e34d"*.csv 2>/dev/null | cut -c1-150
ls -t "$HOME/overnight-scratch/" | head -8
echo
echo "===tail of the newest e34d driver/run log==="
L=$(ls -t "$HOME/overnight-scratch/e34d"*.log 2>/dev/null | head -1)
echo "log: $L"
[ -n "${L:-}" ] && tail -6 "$L"
echo "===DONE==="
