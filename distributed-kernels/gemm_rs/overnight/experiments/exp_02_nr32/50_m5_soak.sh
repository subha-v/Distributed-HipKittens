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
echo "### M5 soak START $(date -u +%FT%TZ)"
setsid -w timeout 3600 docker exec -w $ON/harness dhk-gemmrs python3 m5_soak.py 600 512 4096 12288 1 1 2>&1 | tee $LOG/m5_soak.log
rc=${PIPESTATUS[0]}
echo "### M5 soak END rc=$rc $(date -u +%FT%TZ)"
exit $rc