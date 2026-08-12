#!/usr/bin/env bash
# Transported by tools/nsh.ps1. CPU-only ISA verification of the exp_21 ubench.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
docker exec dhk-gemmrs bash "$EXP/isa.sh"
echo "isa.sh exit=$?"
