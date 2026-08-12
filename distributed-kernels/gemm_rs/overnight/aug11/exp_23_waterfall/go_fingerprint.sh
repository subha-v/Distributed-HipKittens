#!/usr/bin/env bash
# Re-run the fingerprint gate against the already-built rungs. Fast, CPU-only,
# rebuilds nothing -- so it is safe to iterate on the gate itself without
# risking a stale-binary masquerade.
set -uo pipefail
EXP=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall
docker exec -w "$EXP" dhk-gemmrs python3 "$EXP/fingerprint.py"
echo "===== FINGERPRINT EXIT $? ====="
