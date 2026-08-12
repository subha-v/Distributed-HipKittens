#!/usr/bin/env bash
# exp_10 REMATCH -- interleaved same-run paired comparison, OUR kernel (post
# has_bias shape-table fix) vs frozen rank-1, graded protocol, one fresh
# 8-process pool per shape.
#
# Differences from vs_sweep.sh (which produced the now-invalid 1.78x):
#   * VS_FORCE_BIAS defaults to 1. That is what the official evaluator actually
#     does (its cases parser leaves has_bias as the truthy string "False"), and
#     rank-1's cached launch path dereferences bias unconditionally.
#   * fewer iters: we are looking for multiples, not percents.
#   * per-shape wall-clock accounting, so a slow shape is visible while running.
#   * shape order front-loads 5 and 3 (the 8192x8192x29568 and 4096^3 rows that
#     were falling through to generic_config), because those are the sanity
#     check that the fix is live in the .so the harness loads.
#
#   $1 shapes (comma list)  $2 iters  $3 reps  $4 base port  $5 VS_FORCE_BIAS
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
SHAPES=${1:-5,3,0,1,2,4}
ITERS=${2:-12}
REPS=${3:-2}
PORT0=${4:-12700}
VS_FORCE_BIAS=${5:-1}
export VS_FORCE_BIAS
OUT=$ON/experiments/exp_10_rank1/vs_logs

mkdir -p "$OUT"
echo "=== rematch start $(date -Is)  shapes=$SHAPES iters=$ITERS reps=$REPS bias_forced=$VS_FORCE_BIAS"
echo "kfd fds before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
echo "module under test: $(ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so")"

cp "$ON/experiments/exp_10_rank1/mp_vs_rank1.py" "$ARM/mp_vs_rank1.py"

IFS=',' read -ra LIST <<< "$SHAPES"
for s in "${LIST[@]}"; do
  PORT=$((PORT0 + s))
  T0=$(date +%s)
  echo
  echo "################ shape index $s  start $(date -Is) ################"
  rm -f "$ARM"/ipc_handles_rank*.bin
  # wait for the node to drain rather than colliding with a straggler
  for _ in $(seq 1 30); do
    live=$(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)
    [ "$live" = "0" ] && break
    echo "  waiting for kfd drain (fds=$live)"; sleep 5
  done
  docker exec -w "$ARM" \
    -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
    -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
    -e PYTHONUNBUFFERED=1 \
    -e HK_DEBUG=0 \
    -e TRITON_CACHE_DIR="$ARM/.triton" \
    -e HSA_ENABLE_COREDUMP=0 \
    -e AMDGCN_USE_BUFFER_OPS=0 \
    -e VS_FORCE_BIAS="$VS_FORCE_BIAS" \
    -e VS_OUT="$OUT/vs_s${s}" \
    dhk-gemmrs bash -lc "timeout --signal=TERM 900 setsid python3 -u mp_vs_rank1.py $s $ITERS $REPS $PORT" \
    2>&1 | grep -vE '^\[1/|^\[2/|^\[3/|hipcc|^ *[0-9]+ \||warning:|^ *\^|preprocessed|replaced kernel|unsupported CUDA'
  echo "shape $s wall=$(( $(date +%s) - T0 ))s"
done

echo
echo "===== aggregate ====="
python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$OUT"
echo "===== rematch DONE $(date -Is) ====="
