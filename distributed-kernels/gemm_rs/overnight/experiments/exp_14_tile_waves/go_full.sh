#!/usr/bin/env bash
# Rebuild the screening module (row 107 added after draw 1), archive the
# validation draw so it cannot be pooled with a different arm set, then take
# five independent allocation draws and pool them.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
mkdir -p "$D/logs" "$D/draw_archive"

echo "########## archive the validation draw (3 arms, scale 0.3) ##########"
for f in "$D"/sweep_*.json; do
  [ -e "$f" ] || continue
  mv "$f" "$D/draw_archive/validation_$(basename "$f")"
  echo "  archived $(basename "$f")"
done

bash "$D/go_build.sh" || exit 1

bash "$D/run_sweep.sh" 1,2,3 8 1 5
