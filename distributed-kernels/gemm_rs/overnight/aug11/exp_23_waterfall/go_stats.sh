#!/usr/bin/env bash
# Re-score the existing draws with the widened null set. Pure JSON analysis --
# imports no torch, opens no device, so it needs neither the GPU lease nor a
# clean node.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
docker exec -w "$EXP" dhk-gemmrs python3 "$EXP/pool_stats.py" 2>&1 | tee "$EXP/stats.txt"
echo "===== STATS EXIT ${PIPESTATUS[0]} ====="
