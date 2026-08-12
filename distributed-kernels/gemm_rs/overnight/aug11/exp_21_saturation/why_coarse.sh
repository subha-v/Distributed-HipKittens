#!/usr/bin/env bash
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_21_saturation
echo "########## campaign log, coarse phase ##########"
sed -n '/phase: coarse/,$p' "$EXP/logs/campaign_latest.log" | head -60
echo
echo "########## newest sweep log ##########"
latest=$(ls -t "$EXP"/logs/sweep_coarse_*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then echo "-- $latest --"; tail -40 "$latest"; else echo "  (no coarse sweep log)"; fi
echo
echo "########## frozen chain used ##########"
ls -la --time-style=+%H:%M:%S "$(ls -dt "$EXP"/logs/run_* | head -1)"
