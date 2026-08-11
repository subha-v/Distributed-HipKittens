#!/usr/bin/env bash
# CPU-only: the three harness modules at the source's current RELEASE_GROUP,
# plus the frozen pre-E3 golden. Safe to run while another job owns the GPUs.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity
TAG=${1:-build}
mkdir -p "$D/logs"

echo "########## RELEASE_GROUP in the pushed source ##########"
grep -n 'define HK_GEMM_RS_MI300X_RELEASE_GROUP' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp

echo
echo "########## harness modules ##########"
docker exec dhk-gemmrs bash $ON/harness/build.sh 2>&1 | tee "$D/logs/build_${TAG}.log"

echo
echo "########## frozen pre-E3 golden ##########"
docker exec dhk-gemmrs bash $D/build_golden.sh 2>&1 | tee -a "$D/logs/build_${TAG}.log"

echo
grep -qE 'ALL MODULES BUILT' "$D/logs/build_${TAG}.log" && \
  grep -qE 'GOLDEN MODULE BUILT' "$D/logs/build_${TAG}.log" && \
  echo "BUILD_ALL OK" || echo "BUILD_ALL FAILED"
