#!/usr/bin/env bash
# exp_12: re-express the result against the frozen rank-1 submission using the
# SAME instrument that produced the 1.238x figure -- exp_10's mp_vs_rank1.py,
# exp_10's rematch protocol (iters=12 reps=2, VS_FORCE_BIAS=1, one fresh pool
# per shape, arms interleaved with the order reversed every rep), pooling all
# eight ranks. Only VS_OUT differs, so exp_10's logs are not overwritten.
#
#   $1 shapes (comma list)  $2 iters  $3 reps  $4 base port
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
SHAPES=${1:-5,3,0,1,2,4}
ITERS=${2:-12}
REPS=${3:-2}
PORT0=${4:-12800}
OUT=$ON/experiments/exp_12_percall/vs_logs
mkdir -p "$OUT"

echo "=== exp_12 vs rank-1, start $(date -Is) shapes=$SHAPES iters=$ITERS reps=$REPS"
echo "module under test: $(ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so")"
echo "submission.py in sync with hk_submission.py: \
$(cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" && echo yes || echo NO)"
echo "fast path present in the submission under test: \
$(grep -c 'launch_fast' "$ON/harness/submission.py")"

cp "$ON/experiments/exp_10_rank1/mp_vs_rank1.py" "$ARM/mp_vs_rank1.py"

IFS=',' read -ra LIST <<< "$SHAPES"
for s in "${LIST[@]}"; do
  PORT=$((PORT0 + s))
  T0=$(date +%s)
  echo
  echo "################ shape index $s  start $(date -Is) ################"
  rm -f "$ARM"/ipc_handles_rank*.bin
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
    -e VS_FORCE_BIAS=1 \
    -e VS_OUT="$OUT/vs_s${s}" \
    dhk-gemmrs bash -lc "timeout --signal=TERM 900 setsid python3 -u mp_vs_rank1.py $s $ITERS $REPS $PORT" \
    2>&1 | grep -vE '^\[1/|^\[2/|^\[3/|hipcc|^ *[0-9]+ \||warning:|^ *\^|preprocessed|replaced kernel|unsupported CUDA'
  echo "shape $s wall=$(( $(date +%s) - T0 ))s"
done

echo
echo "===== aggregate ====="
python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$OUT"
echo "===== DONE $(date -Is) ====="
