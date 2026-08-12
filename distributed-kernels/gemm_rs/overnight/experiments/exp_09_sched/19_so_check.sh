#!/usr/bin/env bash
# Close the last link: the loaded modules are newer than the shipped source.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
find $ON -name '*.so' -newer $ON/../gemm_rs_mi300x.cpp -printf '%TH:%TM:%TS  %p\n' 2>/dev/null
echo "--- source mtime ---"
stat -c '%y  %n' $ON/../gemm_rs_mi300x.cpp $ON/../gemm_rs_mi300x_hk_adapter.cuh
echo "--- all modules ---"
find $ON -name '*.so' -printf '%TH:%TM:%TS  %10s  %p\n' 2>/dev/null
