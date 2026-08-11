#!/usr/bin/env bash
# Repeatability check: the IDENTICAL M7 command (3 rotations x 50 iters).
# Nothing about the measurement changes; run 1 stays the reported primary.
# Motivated by shape 3's stdev of 8.48 us (rotations 96.44 / 111.53 / 97.28).
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/exp_02_nr32/logs
mkdir -p "$LOG"
cp -f $ON/harness/m7_results.json $LOG/m7_results_run1.json 2>/dev/null && echo "saved run1 json"
pids=$(rocm-smi --showpids 2>&1)
case "$pids" in
  *"No KFD PIDs currently running"*) echo "node clean: no KFD PIDs" ;;
  *) echo "ABORT: KFD PIDs present"; echo "$pids"; exit 90 ;;
esac
echo "### M7 bench RUN2 START $(date -u +%FT%TZ)"
setsid -w timeout 3600 docker exec -w $ON/harness dhk-gemmrs python3 m7_bench.py 3 50 2>&1 | tee $LOG/m7_bench_run2.log
rc=${PIPESTATUS[0]}
cp -f $ON/harness/m7_results.json $LOG/m7_results_run2.json 2>/dev/null && echo "saved run2 json"
echo "### M7 bench RUN2 END rc=$rc $(date -u +%FT%TZ)"
exit $rc