#!/usr/bin/env bash
# exp_10 probe 9 -- NO GPU. Pin the truncation to the store-address expression.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
CACHE=$ARM/.triton
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== A. the epilogue verbatim (1334..1375) ====="
awk 'NR>=1334 && NR<=1375 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== B. the single trunc i64 -> i32 in the LLIR, with context ====="
LL=$(find "$CACHE" -name '_kernel1.llir' | head -1)
echo "file: $LL"
grep -nB4 -A4 'trunc i64 .* to i32' "$LL" | head -40

echo
echo "===== C. the tt.addptr chain feeding the store, in the TTGIR ====="
TG=$(find "$CACHE" -name '_kernel1.ttgir' | head -1)
echo "file: $TG"
echo "--- addptr ops with i64 operands ---"
grep -nE 'tt\.addptr.*i64' "$TG" | head -20
echo "--- the store and the addptr immediately before it ---"
grep -nE 'tt\.addptr|tt\.store' "$TG" | tail -20

echo
echo "===== D. in the TTIR: how is the i64 constant consumed? ====="
TT=$(find "$CACHE" -name '_kernel1.ttir' | head -1)
grep -nE 'c-[0-9]+_i64|tt\.addptr|arith\.trunci|arith\.extsi|tt\.store' "$TT" | tail -30

echo
echo "===== E. does the ISA use 32-bit or 64-bit store addressing? ====="
AM=$(find "$CACHE" -name '_kernel1.amdgcn' | head -1)
echo "file: $AM"
echo "store instruction mix:"
grep -oE '(global|buffer|flat)_store[a-z0-9_]*' "$AM" | sort | uniq -c
echo "===== DONE p9 ====="
