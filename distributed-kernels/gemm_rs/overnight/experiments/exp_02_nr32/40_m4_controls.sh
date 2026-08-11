#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/exp_02_nr32/logs
mkdir -p "$LOG"
pids=$(rocm-smi --showpids 2>&1)
case "$pids" in
  *"No KFD PIDs currently running"*) echo "node clean: no KFD PIDs" ;;
  *) echo "ABORT: KFD PIDs present"; echo "$pids"; exit 90 ;;
esac
echo "### M4 controls START $(date -u +%FT%TZ)"
setsid -w timeout 1800 docker exec -w $ON/harness dhk-gemmrs python3 m4_controls.py 2>&1 | tee $LOG/m4_controls.log
rc=${PIPESTATUS[0]}
echo "### M4 controls END rc=$rc $(date -u +%FT%TZ)"
exit $rc