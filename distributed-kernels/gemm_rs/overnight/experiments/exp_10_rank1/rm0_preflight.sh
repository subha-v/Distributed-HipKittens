#!/usr/bin/env bash
# rematch preflight: is the node free, is the has_bias fix live in the .so the
# harness will load, and are there stale pre-fix JSONs that would poison the
# aggregate?
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1

echo "===== 1. node drain ====="
rocm-smi --showpids 2>&1 | head -20
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
pgrep -af 'mp_vs_rank1|mp_smoke|eval.py' | head

echo
echo "===== 2. build artifacts vs sources (mtime) ====="
ls -l --time-style=full-iso "$ON/harness/build/"*.so 2>&1
echo "--- sources ---"
ls -l --time-style=full-iso "$ON/../gemm_rs_mi300x.cpp" \
  "$ON/../gemm_rs_mi300x_constants.cuh" \
  "$ON/../gemm_rs_mi300x_hk_adapter.cuh" \
  "$ON/../gemm_rs_mi300x_host_abi.hpp" 2>&1

echo
echo "===== 3. what does harness/submission.py import ====="
grep -nE '\.so|build|import |sys\.path|compbench' "$ON/harness/submission.py" | head -30

echo
echo "===== 4. is the has_bias fix in the source that built the .so ====="
grep -rnE 'find_scored_config' "$ON/../gemm_rs_mi300x.cpp" "$ON/../gemm_rs_mi300x_host_abi.hpp" \
  "$ON/../gemm_rs_mi300x_constants.cuh" 2>/dev/null | head
echo "--- context around find_scored_config ---"
grep -rn -A 22 'find_scored_config' "$ON/../gemm_rs_mi300x_host_abi.hpp" 2>/dev/null | head -40

echo
echo "===== 5. other .so copies that could shadow (compbench etc) ====="
find "$ON" -name '*.so' -newermt '2026-01-01' -printf '%T+ %10s %p\n' 2>/dev/null | sort | tail -20

echo
echo "===== 6. stale vs_logs (pre-fix) ====="
ls -l --time-style=full-iso "$ON/experiments/exp_10_rank1/vs_logs/"*.json 2>&1 | head -30
echo "count: $(ls "$ON/experiments/exp_10_rank1/vs_logs/"*.json 2>/dev/null | wc -l)"

echo
echo "===== 7. triton cache (buffer-ops knob) ====="
ls -ld "$ARM/.triton" 2>&1
echo "entries: $(ls "$ARM/.triton" 2>/dev/null | wc -l)"

echo
echo "===== 8. clocks ====="
rocm-smi --showclocks 2>&1 | grep -iE 'sclk|GPU\[0\]' | head -6
echo "===== DONE ====="
