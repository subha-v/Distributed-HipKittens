#!/usr/bin/env bash
# exp_08 gate ladder wrapper: pin clocks first (idle sclk here is ~120 MHz
# against ~1900 under load, and an unpinned short run is unrepeatable by up to
# 60%), then run the mandated ladder unmodified.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -4
bash "$ON/tools/gate_ladder.sh" exp_08_egress
