#!/usr/bin/env bash
# Transported by tools/nsh.ps1. THIS SCRIPT USES ALL EIGHT GPUs.
#
#   nsh.ps1 -Script .../do_sweep.sh -ArgLine "quick"    smoke, ~2-4 min, 1 rotation
#   nsh.ps1 -Script .../do_sweep.sh -ArgLine "full"     the figure data, ~30-60 min
#   nsh.ps1 -Script .../do_sweep.sh -ArgLine "coarse"   payload-granularity cross-check
#
# Node discipline: one 8-GPU job of ours at a time. run_sweep.sh refuses to start if another
# exp_21 sweep is alive; check the node yourself for OTHER agents' jobs before calling this.
set -uo pipefail
MODE=${1:-quick}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation

echo "########## who owns the GPUs right now ##########"
bash "$ON/tools/who_owns_gpus.sh" 2>&1 | tail -20 || rocm-smi 2>&1 | tail -14

echo
echo "########## clocks (pin first if these are not ~1900) ##########"
rocm-smi --showclocks 2>&1 | grep -i sclk || true

echo
echo "########## launching sweep: $MODE ##########"
docker exec dhk-gemmrs bash "$EXP/run_sweep.sh" "$MODE"
rc=$?
echo "run_sweep.sh exit=$rc"

echo
echo "########## artifacts ##########"
ls -la "$EXP"/*.json "$EXP"/logs 2>/dev/null | tail -20
exit $rc
