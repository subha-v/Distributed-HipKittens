#!/usr/bin/env bash
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1
for s in 1 2 3 4; do
  f=$E/vs_logs/vs_s$s.rank0.json
  [ -f "$f" ] || continue
  echo "===== shape $s (rank0) ====="
  cat "$f"
  echo
done
echo "===== DONE ====="
