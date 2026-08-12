#!/usr/bin/env bash
# exp_35 step 9: final state check before releasing the GPU lease.
set -uo pipefail
date -u
echo "===HEAD (must still be ca5b683f)==="
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
echo "--- dirty files (must be empty) ---"
git -C "$HOME/Distributed-HipKittens" status --porcelain | head -10
echo "(end dirty)"

echo "===OUR PROCESSES (must be none)==="
pgrep -af 'torchrun|run_campaign|screen\.sh|e35_' | grep -v 'e35_99' | head -10
echo "(end procs)"

echo "===CAMPAIGN LOCK DIR (must not exist)==="
ls -d /tmp/k0_mok_synthetic_gpu_lock 2>&1

echo "===GPU USE==="
/opt/rocm/bin/rocm-smi --showuse --csv 2>/dev/null | head -12

echo "===KFD PROCESSES==="
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | sed -n '1,20p'

echo "===ARTIFACTS ON NODE==="
ls -la "$HOME/e35/" "$HOME/e35/raw" 2>&1 | head -20
