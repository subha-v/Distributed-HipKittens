#!/usr/bin/env bash
# Render the markdown tables for result.md from stats.json. CPU only.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
docker exec -w "$EXP" dhk-gemmrs python3 "$EXP/table.py" | tee "$EXP/tables.md"
