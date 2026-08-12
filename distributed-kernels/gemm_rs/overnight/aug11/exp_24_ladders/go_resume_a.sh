#!/usr/bin/env bash
# RESUME after the 06:40 preemption. Stage 1: fill in the one missing instrument-A
# shape (index 3, 4096x4096x4096) and aggregate the complete six-shape ladder.
#
# LAD_KEEP=1   do NOT wipe the five surviving shapes
# LAD_SHAPES=3 measure only the missing one
# LAD_EXPECT_A=6 but still assert all six are present in the aggregate
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
export LAD_KEEP=1 LAD_SHAPES=3 LAD_EXPECT_A=6 LAD_PHASE=A LAD_CAP=2400
exec bash "$D/go_full.sh"
