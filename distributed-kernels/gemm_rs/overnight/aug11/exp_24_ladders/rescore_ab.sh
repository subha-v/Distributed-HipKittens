#!/usr/bin/env bash
# Re-score the paired campaign with the order-balanced analysis added. CPU only,
# no lease, no GPU. Run via nsh.ps1 alone -- do NOT push.ps1, which syncs *.json
# and would overwrite node-side results with stale local copies.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders
python3 "$D/ab_prev_report.py" > "$D/ab_prev_report.txt" 2>&1
rc=$?
sed -n '/paired deltas/,$p' "$D/ab_prev_report.txt"
echo "report rc=$rc  ab_prev.json $(stat -c%s "$D/ab_prev.json" 2>/dev/null) B"
