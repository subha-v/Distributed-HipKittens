#!/usr/bin/env bash
# CPU-only. No GPU is touched, so this is safe to run at any time.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
docker exec -w "$ON/experiments/exp_08_egress" dhk-gemmrs python3 destmap.py
