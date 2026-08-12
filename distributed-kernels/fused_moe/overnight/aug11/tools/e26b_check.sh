#!/usr/bin/env bash
# exp_26 follow-up step 0: confirm the immutable snapshot from the first run is
# still intact, so every mask build shares ONE base with the original B0/B1/B2.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
DHK=$SC/dhk

echo "===SNAPSHOT_COMMIT==="
git -C "$DHK" rev-parse HEAD 2>&1
git -C "$DHK" status --porcelain 2>&1 | head -20
echo "===DONOR==="
sha256sum "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/n2_phase1_gm.cpp"
echo "===PRIOR_ARTIFACTS==="
ls -la "$SC/out/" 2>&1 | head -40
echo "===PRIOR_TU==="
ls -la "$SC/tu/b0" "$SC/tu/b1" 2>&1
echo "===GPU_IDLE_CHECK (we are CPU-only; just recording who else is up)==="
pgrep -af 'torchrun|mpirun' | head -5
echo "(pgrep done)"
echo "===CONTAINER==="
docker ps --format '{{.Names}}' | grep -c subha_k1
