#!/usr/bin/env bash
# Read-only: how does eval.py drive ranks, groups and warmup? Needed to know
# how many process groups a run creates and where a hang can sit.
set -uo pipefail
DIR=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/ours
E=$DIR/eval.py
U=$DIR/utils.py

echo "===== init/destroy/barrier/Pool sites in eval.py ====="
grep -n 'init_process_group\|destroy_process_group\|barrier\|Pool\|apply_async\|imap\|starmap\|warmup\|recheck\|set_device\|maxtasksperchild\|timeout' $E 2>&1 | head -60

echo
echo "===== _run_distributed_benchmark and friends ====="
sed -n '/def _run_distributed/,/^def [a-z_]*(/p' $E 2>&1 | head -90

echo
echo "===== the per-rank entry that calls custom_kernel ====="
grep -n 'custom_kernel' $E $U 2>&1 | head -20

echo
echo "===== READ DONE ====="
