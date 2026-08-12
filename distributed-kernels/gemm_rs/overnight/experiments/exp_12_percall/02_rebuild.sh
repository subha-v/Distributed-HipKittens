#!/usr/bin/env bash
# Guarantee build/gemm_rs_mi300x.so matches the source of record before any
# number is attributed to it, and record the resource tuple.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
docker exec dhk-gemmrs bash "$ON/harness/build.sh" 2>&1 | tail -20
echo "m1_rc=$?"
docker exec dhk-gemmrs bash "$ON/tools/m2_report.sh" 2>&1 | tail -30
echo "m2_rc=$?"
ls -la "$ON/harness/build/gemm_rs_mi300x.so"
echo "DONE-rebuild"
