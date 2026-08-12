#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
echo "############### ROUND 3 ONLY (shuffled arm order) ###############"
cat "$D/logs/paired_r3.txt"
echo
echo "############### round 3 per-shape summaries ###############"
for f in ab3_fwd ab3_rev; do
  echo "===== $f ====="
  sed -n '/PER-SHAPE SUMMARY/,/^wrote/p' "$D/logs/$f.log"
done
