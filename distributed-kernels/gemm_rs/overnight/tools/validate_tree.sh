#!/usr/bin/env bash
# Prove the relocated overnight tree still builds and runs, so the handoff
# starts from a known-good state rather than a hopeful one.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== tree present? ====="
ls "$ON" "$ON/harness" | head -30

echo
echo "===== build all three modules ====="
docker exec dhk-gemmrs bash "$ON/harness/build.sh" 2>&1 | tail -8

echo
echo "===== pre-launch smoke (no kernel) ====="
docker exec -w "$ON/harness" dhk-gemmrs timeout 300 python3 -u smoke.py 2>&1 | tail -12

echo
echo "===== single-process correctness, one graded shape ====="
docker exec -w "$ON/harness" dhk-gemmrs timeout 600 python3 -u m3_correctness.py one 512 4096 12288 1 3 2>&1 | tail -10

echo
echo "===== DONE ====="
