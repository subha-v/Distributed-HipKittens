#!/usr/bin/env bash
set -u
SRC=/home/subvadla/dhk/distributed-kernels/gemm_rs
ON=$SRC/overnight
echo "=== sha256 kernel sources on node NOW ==="
sha256sum $SRC/gemm_rs_mi300x_host_abi.hpp $SRC/gemm_rs_mi300x.cpp $SRC/gemm_rs_mi300x_constants.cuh $SRC/gemm_rs_mi300x_hk_adapter.cuh
echo
echo "=== mtimes of kernel sources on node ==="
stat -c "%y  %n" $SRC/gemm_rs_mi300x_host_abi.hpp $SRC/gemm_rs_mi300x.cpp $SRC/gemm_rs_mi300x_constants.cuh $SRC/gemm_rs_mi300x_hk_adapter.cuh
echo
echo "=== mtimes of built modules (M1 output actually used by M3-M7) ==="
stat -c "%y  %n" $ON/harness/build/*.so
echo
echo "=== scored_shapes NR column on node NOW ==="
sed -n "53,60p" $SRC/gemm_rs_mi300x_host_abi.hpp