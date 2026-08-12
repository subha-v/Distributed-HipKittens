#!/usr/bin/env bash
# nsh entry point: run the CPU-only ISA census inside the compile container.
# No GPU is touched: `hipcc -c` is a host-side compile.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_27_mainloop
docker exec dhk-gemmrs bash "$EXP/isa_census.sh" 2>&1 | tee "$EXP/census.log"
echo "launcher exit=${PIPESTATUS[0]}"
