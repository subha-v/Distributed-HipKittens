#!/usr/bin/env bash
# aug11/exp_20: build counters.json from prof/*/p_counter_collection.csv, and
# keep the aggregate text log as counters.log. CPU-only.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution

cp -f "$OUT/counters_run.log" "$OUT/counters.log" 2>/dev/null || true

docker exec -w "$OUT" dhk-gemmrs python3 mk_counters_json.py "$OUT/prof" \
  "$OUT/counters.json" 2 4

echo
echo "===== validation ====="
docker exec -w "$OUT" dhk-gemmrs python3 -c \
  "import json; d=json.load(open('counters.json')); print('cells:', len(d['cells'])); print('unusable:', [t for t,c in d['cells'].items() if not c.get('usable')])"
