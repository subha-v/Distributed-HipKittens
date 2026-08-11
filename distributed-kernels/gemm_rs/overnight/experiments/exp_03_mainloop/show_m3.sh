#!/usr/bin/env bash
set -uo pipefail
ARM=${1:?usage: show_m3.sh <arm>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$ON/experiments/exp_03_mainloop/arms/$ARM
echo "########## m3_correctness.log (last 90 lines) ##########"
tail -90 "$L/m3_correctness.log"
