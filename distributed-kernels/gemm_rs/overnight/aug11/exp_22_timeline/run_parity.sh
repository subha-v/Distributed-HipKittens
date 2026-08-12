#!/usr/bin/env bash
# Transport wrapper: run exp_22's parity gate inside dhk-gemmrs.
# CPU-only (three hipcc compiles); it does not touch the GPU and does not need
# the node lease.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline

echo "===== node-side source state ====="
grep -n 'define HK_GEMM_RS_MI300X_TRACE' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
grep -c 'HK_TRACE_STAMP(' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp

echo
echo "===== parity gate ====="
docker exec -w "$E22" dhk-gemmrs bash "$E22/parity_gate.sh"
echo "parity_gate exit=$?"
