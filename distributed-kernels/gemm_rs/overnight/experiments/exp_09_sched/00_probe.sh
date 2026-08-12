#!/usr/bin/env bash
# exp_09_sched step 0: node state + a pristine snapshot of the validated-best
# sources + the "before" schedule shape. NO GPU work, NO edits.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
mkdir -p "$EXP/arms" "$EXP/logs"

echo "########## node ##########"
date
rocm-smi --showpids 2>&1 | head -20
echo "KFD pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
docker ps --format '{{.Names}}\t{{.Status}}' | head

echo
echo "########## clocks ##########"
rocm-smi --showclocks 2>&1 | grep -iE 'sclk|GPU\[0\]' | head -6

echo
echo "########## pristine restore point (validated best) ##########"
mkdir -p "$EXP/base"
cp -f "$ON/../gemm_rs_mi300x.cpp"            "$EXP/base/"
cp -f "$ON/../gemm_rs_mi300x_hk_adapter.cuh" "$EXP/base/"
sha256sum "$EXP/base/"*

echo
bash "$EXP/isa_arm.sh" base
