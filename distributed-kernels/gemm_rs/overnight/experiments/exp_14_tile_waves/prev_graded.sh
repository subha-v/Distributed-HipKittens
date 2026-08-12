#!/usr/bin/env bash
# Re-report exp_13's stored graded samples so the before/after per-shape table
# uses ONE reporting path for both sessions. No GPU: this only re-reads JSON.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
for d in "$ON"/experiments/exp_13_cta_split/vs_logs/*/; do
  echo "########## $(basename "$d") ##########"
  python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$d" 2>&1 | \
    sed -n '/^ #/,/GEOMETRIC/p;/geo   best/p;/geo median/p'
done
