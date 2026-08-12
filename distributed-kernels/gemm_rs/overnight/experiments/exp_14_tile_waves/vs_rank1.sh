#!/usr/bin/env bash
# The graded number for whatever is currently built into
# harness/build/gemm_rs_mi300x.so, measured against the frozen rank-1
# submission with exp_10's instrument and rematch protocol: 8 processes, the
# evaluator's own barrier -> call -> synchronize -> barrier region, one fresh
# pool per shape, the two arms interleaved with the order reversed every rep,
# all eight ranks pooled.
#
# Copied verbatim from exp_13_cta_split/vs_rank1.sh with D redirected here.
# NOTHING else differs -- the instrument must stay byte-comparable to the run
# that produced exp_13's 348.64/306.60 pair, since that pair is the denominator
# this experiment is measured against. The provenance dump already prints the
# whole scored_shapes table, which is exactly the column exp_14 changes.
#
# Why the graded protocol and not our pipelined harness: rank-1 is measured in
# the SAME run, so a globally fast or slow run shows up in the anchor rather
# than in our number, and the graded geomean is the competition's actual ranking
# statistic. It also exposes the reduce tail that pipelining hides, which is
# exactly the mechanism exp_13 moves.
#
#   vs_rank1.sh <tag> [shapes_csv] [iters] [reps] [base_port]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_14_tile_waves
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
TAG=${1:?usage: vs_rank1.sh <tag> [shapes] [iters] [reps] [port0]}
SHAPES=${2:-5,3,0,1,2,4}
ITERS=${3:-12}
REPS=${4:-2}
PORT0=${5:-12900}
OUT=$D/vs_logs/$TAG
mkdir -p "$OUT" "$D/logs"

echo "=== exp_14 vs rank-1 [$TAG], start $(date -Is) shapes=$SHAPES iters=$ITERS reps=$REPS"
echo "--- BM/BN/BK + reducer-split table in the source that built it ---"
sed -n '/inline constexpr std::array<shape_entry, 6> scored_shapes/,/}};/p' \
  $ON/../gemm_rs_mi300x_host_abi.hpp
echo "module under test: $(ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so")"
echo "runtime under test: $(ls -l --time-style=full-iso "$ON/harness/build/dhk_rt.so")"
echo -n "submission.py in sync with hk_submission.py: "
cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" && echo yes || echo NO
echo "fast path present in the submission under test: $(grep -c 'launch_fast' "$ON/harness/submission.py")"

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
echo "===== aggregate [$TAG] ====="
python3 "$ON/experiments/exp_10_rank1/vs_report.py" "$OUT"
echo "===== DONE $TAG $(date -Is) ====="
