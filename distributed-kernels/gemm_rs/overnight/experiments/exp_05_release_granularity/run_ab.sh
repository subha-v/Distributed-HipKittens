#!/usr/bin/env bash
# Build the three RELEASE_GROUP arm modules and run the paired A/B. 8-GPU job,
# so it refuses a dirty node and waits for one that is taken.
#
#   run_ab.sh [rounds] [iters] [tag]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity
ROUNDS=${1:-5}
ITERS=${2:-50}
TAG=${3:-ab}
REVERSE=${4:-0}
mkdir -p "$D/logs"

kfd_pids() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l; }

docker exec dhk-gemmrs bash $D/build_arms.sh 1 2 4 4c 1b 2>&1 | tee "$D/logs/build_arms.log"
grep -q 'ARM MODULES BUILT' "$D/logs/build_arms.log" || { echo "AB ABORTED: arm build failed"; exit 1; }

for i in $(seq 0 179); do
  [ "$(kfd_pids)" = "0" ] && break
  [ $((i % 6)) = 0 ] && echo "  waiting for the node: $(kfd_pids) KFD pids ($((i * 20))s)"
  sleep 20
done
[ "$(kfd_pids)" = "0" ] || { echo "AB REFUSED: node never went clean"; exit 1; }

docker exec -w $ON/harness -e AB_TAG="$TAG" dhk-gemmrs timeout 5400 \
  python3 -u $D/ab_release_group.py "$ROUNDS" "$ITERS" 600 "$REVERSE" 2>&1 \
  | tee "$D/logs/ab_${TAG}.log"
status=${PIPESTATUS[0]}
echo "ab exit status: $status  (log: $D/logs/ab_${TAG}.log)"
exit $status
