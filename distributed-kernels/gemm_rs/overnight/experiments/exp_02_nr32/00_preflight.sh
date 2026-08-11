#!/usr/bin/env bash
# exp_02_nr32 preflight: node cleanliness, container, clocks, source-sync check.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRC=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "=== host / date ==="
hostname
date -u +%Y-%m-%dT%H:%M:%SZ

echo
echo "=== docker ps ==="
docker ps --format "{{.Names}} | {{.Status}}"

echo
echo "=== rocm-smi --showpids ==="
rocm-smi --showpids

echo
echo "=== our stray processes ==="
ps -eo pid,user,etime,cmd | grep -E "mp_smoke|eval\.py|torchrun|m7_bench|m5_soak|m3_correctness|m4_controls" | grep -v grep
echo "(end stray list, rc=$?)"

echo
echo "=== scored_shapes on node ==="
sed -n "50,64p" $SRC/gemm_rs_mi300x_host_abi.hpp

echo
echo "=== sha256 kernel sources on node ==="
sha256sum $SRC/gemm_rs_mi300x_host_abi.hpp $SRC/gemm_rs_mi300x.cpp $SRC/gemm_rs_mi300x_constants.cuh $SRC/gemm_rs_mi300x_hk_adapter.cuh

echo
echo "=== sclk levels (pinned?) ==="
rocm-smi --showclocks

echo
echo "=== setsid -w available? ==="
setsid -w true && echo "setsid -w OK"

echo
echo "=== harness dir ==="
ls -la $ON/harness