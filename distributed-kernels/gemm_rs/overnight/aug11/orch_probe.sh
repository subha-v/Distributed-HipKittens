#!/usr/bin/env bash
# Orchestrator's read-only node probe. Answers only: is a GPU job running, whose
# is it, and what has landed on the node so far. Touches nothing.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== GPU occupancy ====="
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pid count: $n"

echo
echo "===== our processes ====="
ps -eo pid,etime,stat,cmd 2>/dev/null | grep -E 'exp_ablation|m7_bench|m3_correctness|reattribute|sweep|saturation|eval\.py' | grep -v grep

echo
echo "===== aug11 experiment dirs on the node ====="
for d in "$ON"/aug11/*/; do
  [ -d "$d" ] || continue
  printf '%-34s %s files\n' "$(basename "$d")" "$(find "$d" -type f | wc -l)"
done

echo
echo "===== ablation arm .so freshness ====="
ls -la "$ON"/harness/build/gemm_rs_abl_*.so 2>&1
echo "-- generated scratch --"
ls -la "$ON"/harness/ablate/ 2>&1

echo
echo "===== newest reattribute log ====="
ls -t "$ON"/experiments/logs/reattribute_*.log 2>/dev/null | head -3
echo "done"
