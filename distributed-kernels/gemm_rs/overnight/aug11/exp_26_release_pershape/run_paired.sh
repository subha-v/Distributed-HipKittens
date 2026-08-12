#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
python3 "$D/paired.py" "$D/logs" 2>&1 | tee "$D/logs/paired.txt"
