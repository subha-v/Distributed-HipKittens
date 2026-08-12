#!/usr/bin/env bash
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
docker exec dhk-gemmrs bash -lc "cd $D && python3 report_b.py" 2>&1
echo
echo "===== raw popcorn heads, to confirm the labels are right this time ====="
for f in ours/benchmark reference/benchmark rank1/bench rank1/warm; do
  echo "--- rot0/$f.popcorn.txt"
  sed -n '1,5p' "$D/raw/eval/rot0/$f.popcorn.txt" 2>/dev/null || echo "   MISSING"
done
echo
echo "===== sha256 of each captured popcorn (all must differ) ====="
find "$D/raw/eval" -name '*.popcorn.txt' | sort | while read -r f; do
  echo "  $(sha256sum "$f" | cut -c1-16)  ${f#$D/raw/eval/}"
done
