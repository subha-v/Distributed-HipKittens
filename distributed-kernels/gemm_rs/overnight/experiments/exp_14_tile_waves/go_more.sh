#!/usr/bin/env bash
# Five more independent draws on top of the first five, then re-pool.
# Shape 1's candidate is the marginal one -- its effect (~2%) sits under the
# node's between-process drift -- so it is the reason for doubling the evidence.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
bash "$D/run_sweep.sh" 1,2,3 8 1 5 || exit 1
docker exec -w $D dhk-gemmrs python3 -u pool.py 2>&1 | tee "$D/logs/pool.log"
echo "POOL rc=${PIPESTATUS[0]}"
