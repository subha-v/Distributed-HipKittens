#!/usr/bin/env bash
# Host-side launcher: build_sweep.sh needs the container's ROCm/pybind toolchain.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
mkdir -p "$D/logs"
docker exec dhk-gemmrs bash "$D/build_sweep.sh" 2>&1 | \
  tee "$D/logs/build_sweep.log"
echo "BUILD rc=${PIPESTATUS[0]}"
