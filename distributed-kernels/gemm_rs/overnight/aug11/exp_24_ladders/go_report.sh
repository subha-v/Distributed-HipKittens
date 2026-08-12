#!/usr/bin/env bash
# Aggregate the now-complete six-shape instrument-A ladder and print the full
# report. No GPU work, so no lease needed.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders

echo "===== rank files present ====="
for i in 0 1 2 3 4 5; do
  echo "  shape idx $i : $(ls "$D/raw/ladder"/lad_s${i}.rank*.json 2>/dev/null | wc -l)/8"
done

echo
echo "===== aggregate (strict: all six shapes, instrument B not yet run) ====="
docker exec dhk-gemmrs bash -lc \
  "cd $D && python3 ladders.py --root $D --out $D/ladders.json --ladder-dir ladder --expect-a 6 --expect-b 0" 2>&1 | tail -30
cp -f "$D/ladders.json" "$D/ladders_instrumentA.json"

echo
echo "===== FULL REPORT ====="
docker exec dhk-gemmrs bash -lc "cd $D && python3 report.py $D/ladders.json" 2>&1
