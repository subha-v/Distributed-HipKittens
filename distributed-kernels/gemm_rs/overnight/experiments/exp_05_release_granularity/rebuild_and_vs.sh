#!/usr/bin/env bash
# Rebuild the production module from the pushed source, then take the graded
# number for it. One remote round trip instead of three.
#
#   rebuild_and_vs.sh <tag> [shapes_csv] [iters] [reps] [base_port]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity
TAG=${1:?usage: rebuild_and_vs.sh <tag> [shapes] [iters] [reps] [port0]}

mkdir -p "$D/logs"
docker exec dhk-gemmrs bash $ON/harness/build.sh > "$D/logs/build_${TAG}.log" 2>&1
tail -8 "$D/logs/build_${TAG}.log"
grep -q 'ALL MODULES BUILT' "$D/logs/build_${TAG}.log" || {
  echo "REBUILD FAILED for $TAG"; exit 1; }

bash $D/vs_rank1.sh "$TAG" "${2:-5,3,0,1,2,4}" "${3:-12}" "${4:-2}" "${5:-12900}"
