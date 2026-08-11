#!/usr/bin/env bash
# Tonight's ratchet baseline: our harness, six graded shapes, on a verified
# clean node. Always re-establish this in-session -- the recorded 285.7 us came
# from a different session and a node whose state we cannot re-verify.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOG=$ON/experiments/logs
mkdir -p "$LOG"
stamp=$(date -u +%Y%m%dT%H%M%SZ)

echo "===== pre-flight: node must be clean ====="
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
if [ "$n" != "0" ]; then echo "ABORT: node dirty, run reap_stale.sh first"; exit 1; fi

echo "===== clocks ====="
rocm-smi 2>&1 | tail -12 | head -10

echo "===== m7_bench 3 rotations x 50 iters ====="
docker exec -w $ON/harness dhk-gemmrs \
  timeout 1800 python3 -u m7_bench.py 3 50 2>&1 | tee "$LOG/baseline_$stamp.log"

echo "===== log at $LOG/baseline_$stamp.log ====="
echo "===== DONE ====="
