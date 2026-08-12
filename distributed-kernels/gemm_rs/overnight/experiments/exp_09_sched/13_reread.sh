#!/usr/bin/env bash
# Re-read an already-archived arm .s with the current sched_isa.py. No build.
#   13_reread.sh <arm> [prefix]
set -uo pipefail
ARM=${1:?arm}; WANT=${2:-'<256,256,32'}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_09_sched
D=$EXP/arms/$ARM
python3 "$EXP/sched_isa.py" "$D/gemm_rs_mi300x.gfx942.s" "$WANT" | tee "$D/kloop_256.txt"
