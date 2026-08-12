#!/usr/bin/env bash
# exp_26 step 3b: unbundle the genco output, then disassemble.
set -uo pipefail
docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
echo "=== FILE MAGIC ==="
file $SC/out/B0.hsaco
head -c 32 $SC/out/B0.hsaco | od -c | head -3
echo "=== BUNDLE LIST ==="
/opt/rocm/llvm/bin/clang-offload-bundler --type=o --list --input=$SC/out/B0.hsaco 2>&1 | head
'
