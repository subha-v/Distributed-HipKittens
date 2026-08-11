#!/usr/bin/env bash
# exp_01 step 0: node must be clean, and record the node-side kernel source
# hashes BEFORE any push so a stale Windows copy cannot silently revert the
# validated kernel.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
KS=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "===== host / date ====="
hostname
date -u +%Y-%m-%dT%H:%M:%SZ

echo
echo "===== rocm-smi --showpids ====="
rocm-smi --showpids 2>&1 | tail -30

echo
echo "===== our stale processes? ====="
pgrep -a -f 'eval.py|mp_smoke|m7_bench|m3_correctness' 2>&1 | head -20 || echo "none"

echo
echo "===== containers ====="
docker ps --format '{{.Names}}\t{{.Status}}' 2>&1 | head -10

echo
echo "===== node kernel source hashes ====="
sha256sum $KS/gemm_rs_mi300x.cpp $KS/gemm_rs_mi300x_hk_adapter.cuh \
          $KS/gemm_rs_mi300x_constants.cuh $KS/gemm_rs_mi300x_host_abi.hpp 2>&1

echo
echo "===== node hk_submission.py hash + harness build ====="
sha256sum $ON/harness/hk_submission.py 2>&1
ls -la $ON/harness/build/*.so 2>&1 | head -10

echo
echo "===== clocks ====="
rocm-smi --showclocks 2>&1 | grep -i -m4 'sclk' || true

echo
echo "===== PREFLIGHT DONE ====="
