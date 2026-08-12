#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
docker exec -w $D dhk-gemmrs python3 -u static_probe.py 2>&1 | \
  tee "$D/logs/static_probe.log"
echo "PROBE rc=${PIPESTATUS[0]}"
