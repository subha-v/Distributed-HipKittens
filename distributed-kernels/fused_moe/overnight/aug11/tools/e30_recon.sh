#!/usr/bin/env bash
# exp_30 stage 1 recon: node state, hkp_sync grid_barrier source, launcher tools.
set -uo pipefail
echo "=== host / date ==="
hostname; date -u
echo
echo "=== GPU users (must be gpuagent only) ==="
/opt/rocm/bin/rocm-smi --showpids 2>&1 | head -30
echo
echo "=== torchrun/mpirun ==="
pgrep -af 'torchrun|mpirun' || echo "(none)"
echo
echo "=== node DHK checkout ==="
git -C "$HOME/Distributed-HipKittens" log --oneline -1
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
git -C "$HOME/Distributed-HipKittens" rev-parse --abbrev-ref HEAD
git -C "$HOME/Distributed-HipKittens" status --porcelain=v1 | head -20
echo
echo "=== tools on node ==="
ls -la "$HOME/tools" 2>&1 | head -40
echo
echo "=== hkp_sync.hpp locations ==="
find "$HOME/amd-master" -name 'hkp_sync.hpp' 2>/dev/null | head -5
echo
echo "=== grid_barrier / local_grid_epoch ==="
F=$(find "$HOME/amd-master" -name 'hkp_sync.hpp' 2>/dev/null | head -1)
if [ -n "$F" ]; then
  echo "file: $F"
  grep -n 'local_grid_epoch\|grid_barrier\|fail_closed\|struct ' "$F" | head -60
fi
echo
echo "=== lock dir ==="
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 || echo "(no lock)"
echo
echo "=== free disk ==="
df -h "$HOME" | tail -1
