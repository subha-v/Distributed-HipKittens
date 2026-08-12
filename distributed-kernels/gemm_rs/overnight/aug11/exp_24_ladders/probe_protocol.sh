#!/usr/bin/env bash
# READ-ONLY: the exact multi-GPU graded timed region in eval.py, so ladder_mp.py
# can reproduce it verbatim and design.md can describe the harness constant
# honestly rather than from memory.
set -uo pipefail
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
echo "########## eval.py 310..410 (multi-GPU benchmark) ##########"
sed -n '310,412p' "$SRC/eval.py"
echo
echo "########## eval.py 240..262 (warmup / entry) ##########"
sed -n '240,262p' "$SRC/eval.py"
echo
echo "########## reference.py generate_input + check ##########"
sed -n '1,71p' "$SRC/reference.py"
echo "########## DONE ##########"
