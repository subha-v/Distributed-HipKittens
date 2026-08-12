#!/usr/bin/env bash
# Shape 3's tail is synchronized across all 8 ranks at the same sample indices
# (16, 19, 23) with magnitudes matching to under 1%, so it is a collective-level
# stall, not per-rank jitter -- and it did not exist in the pre-fix run of the
# same shape. Re-run shape 3 ALONE into a separate output dir to see whether it
# reproduces. Writes to vs_logs_recheck3 so the primary result is untouched.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
OUT=$E/vs_logs_recheck3
mkdir -p "$OUT"
rm -f "$OUT"/* "$ARM"/ipc_handles_rank*.bin

cat > /tmp/recheck3.sh <<'EOS'
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
{
docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 -e HK_DEBUG=0 \
  -e TRITON_CACHE_DIR="$ARM/.triton" -e HSA_ENABLE_COREDUMP=0 \
  -e AMDGCN_USE_BUFFER_OPS=0 -e VS_FORCE_BIAS=1 \
  -e VS_OUT="$E/vs_logs_recheck3/vs_s2" \
  dhk-gemmrs bash -lc "timeout --signal=TERM 600 setsid python3 -u mp_vs_rank1.py 2 12 2 12811" \
  2>&1 | grep -E 'correctness|best=|exit codes'
echo "recheck done $(date -Is)"
} > "$E/recheck3.log" 2>&1
EOS

setsid nohup bash /tmp/recheck3.sh < /dev/null > /dev/null 2>&1 &
echo "detached recheck pid=$!"
echo "===== DONE ====="
