#!/usr/bin/env bash
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -20
echo "--- acquire loop output so far ---"
ps -eo pid,etimes,cmd | grep -E "gpu_lease|go_gpu" | grep -v grep | head
echo "--- lease dir ---"
ls -ld /tmp/*lease* /tmp/dhk* 2>/dev/null | head
echo "--- who is on the GPUs ---"
rocm-smi --showpids 2>/dev/null | head -12
