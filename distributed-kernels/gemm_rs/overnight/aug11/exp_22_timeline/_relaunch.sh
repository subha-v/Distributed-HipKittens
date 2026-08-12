#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
echo "===== relaunching arms: ${EXP22_ARMS:-all} ====="
EXP22_REUSE_M7=1 EXP22_ARMS="${EXP22_ARMS:-all}" bash "$E22/_launch.sh"
