#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
docker exec -w $D dhk-gemmrs python3 -u pool.py 2>&1 | tee "$D/logs/pool.log"
echo "POOL rc=${PIPESTATUS[0]}"
