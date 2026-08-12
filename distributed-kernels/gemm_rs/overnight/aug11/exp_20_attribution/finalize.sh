#!/usr/bin/env bash
# Enrich both deliverables in place. No GPU work.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
sed -i 's/\r$//' "$OUT"/*.py 2>/dev/null

CLK=$(rocm-smi --showperflevel 2>/dev/null | grep -c perf_determinism)

echo "===== per-shape rollup into counters.json ====="
python3 "$OUT/rollup_counters.py" "$OUT/counters.json" "$OUT/ablation.json"

echo
echo "===== geometry + xSOL + M7 into ablation.json ====="
python3 "$OUT/add_geometry.py" "$OUT/ablation.json" "$OUT/m7_results.json" \
  "clocks_before=perf_determinism on $CLK/8 GPUs" \
  "clocks_after=perf_determinism on $CLK/8 GPUs"

echo
echo "===== M7 shape-2 outlier check (its stdev was 27.15) ====="
python3 - "$OUT/m7_results.json" << 'PY'
import json, sys, statistics
d = json.load(open(sys.argv[1]))
for i in range(6):
    s = d["samples_us"][str(i)]
    print(f"  shape {i+1}: samples {[round(x,2) for x in s]} "
          f"min={min(s):.2f} median={statistics.median(s):.2f} mean={sum(s)/len(s):.2f}")
PY

echo
echo "===== deliverables ====="
ls -la "$OUT"/*.json
