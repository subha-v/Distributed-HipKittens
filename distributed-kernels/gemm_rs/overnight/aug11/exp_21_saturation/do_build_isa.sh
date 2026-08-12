#!/usr/bin/env bash
# Transported by tools/nsh.ps1. Rebuild + re-verify ISA. Compilation only, no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation
md5sum "$EXP/sat_ubench.cpp"
docker exec dhk-gemmrs bash "$EXP/build.sh" 2>&1 | tail -25
echo "=================================================================="
docker exec dhk-gemmrs bash "$EXP/isa.sh" 2>&1 | sed -n '/opcode census/,/amdhsa metadata/p;/SHIPPED/,$p'
echo "isa exit=$?"
echo
echo "########## python syntax check on the driver (no GPU, no import of the .so) ##########"
docker exec dhk-gemmrs python3 -m py_compile "$EXP/run_saturation.py" && echo "run_saturation.py compiles"
