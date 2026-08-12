#!/usr/bin/env bash
# exp_22 pre-flight before taking the GPU lease. Read-only: it starts nothing.
set -uo pipefail
echo "===UTC==="; date -u +%FT%TZ; hostname

echo "===FOREIGN_LAUNCHERS (must be empty)==="
pgrep -af 'torchrun|mpirun|e004pf|run_campaign' || echo "(none)"

echo "===ROCM_SMI_SHOWPIDS (expect only gpuagent)==="
/opt/rocm/bin/rocm-smi --showpids 2>&1

echo "===STALE_LOCK (must not exist)==="
ls -l /tmp/k0_mok_synthetic_gpu_lock 2>&1 || echo "(absent)"

echo "===UTILISATION==="
/opt/rocm/bin/rocm-smi --showuse 2>&1 | head -30

echo "===VRAM==="
/opt/rocm/bin/rocm-smi --showmemuse 2>&1 | head -30

echo "===OUR_LEFTOVERS (must be empty)==="
pgrep -af e22_saturation || echo "(none)"

echo "===BINARY==="
ls -l --time-style=+%FT%TZ /home/subvadla/e22/out/e22_saturation
sha256sum /home/subvadla/e22/out/e22_saturation
grep -m1 'define E22_SRC_REV' /home/subvadla/e22/src/e22_saturation.hip
echo "===DONE==="
