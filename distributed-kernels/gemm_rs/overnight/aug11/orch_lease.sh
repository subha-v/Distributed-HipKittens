#!/usr/bin/env bash
# Orchestrator: check the GPU lease and node state in one call.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash "$ON/tools/gpu_lease.sh" status
echo
echo "--- recent lease log ---"
tail -12 "$ON/aug11/gpu_lease.log" 2>/dev/null || echo "(no log yet)"
echo
echo "--- our GPU-capable processes ---"
ps -eo pid,etime,stat,cmd 2>/dev/null | grep -E 'ladder_mp|eval\.py|sweep\.py|run_saturation|m7_bench|exp_ablation|trace_run' | grep -v grep || echo "(none)"
