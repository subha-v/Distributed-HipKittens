#!/usr/bin/env bash
# Analyse the already-captured per-iteration series for bimodality. No GPU.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_07_cold_l2

echo "===== what series do we have? ====="
ls -la $E/logs/*.json | head -30

echo
echo "===== bimodality scan ====="
docker exec dhk-gemmrs python3 $E/bimodal.py $E/logs 2>&1 | tail -160
echo "===== DONE ====="
