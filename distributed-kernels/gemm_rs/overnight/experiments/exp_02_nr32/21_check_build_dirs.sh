#!/usr/bin/env bash
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
echo "=== $ON/build ==="
ls -la $ON/build 2>&1
echo
echo "=== $ON/build/isa ==="
ls -la $ON/build/isa 2>&1
echo
echo "=== $ON/harness/build ==="
ls -la $ON/harness/build 2>&1