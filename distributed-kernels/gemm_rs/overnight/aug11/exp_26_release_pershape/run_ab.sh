#!/usr/bin/env bash
# The paired A/B, run in BOTH allocation orders. 8-GPU job: refuses a dirty node
# and waits for one that is taken.
#
#   run_ab.sh [rounds] [iters] [tag]
#
# Two passes, forward and reversed. The harness bias on this node is
# per-allocation and partly allocation-ORDER, so a candidate delta only counts
# if it survives both directions and clears each shape's own null arm.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
ROUNDS=${1:-5}
ITERS=${2:-50}
TAG=${3:-ab}
mkdir -p "$D/logs"

kfd_pids() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l; }

wait_clean() {
  for i in $(seq 0 179); do
    [ "$(kfd_pids)" = "0" ] && return 0
    [ $((i % 6)) = 0 ] && echo "  waiting for the node: $(kfd_pids) KFD pids ($((i * 20))s)"
    sleep 20
  done
  return 1
}

echo "########## clocks (must be pinned before any timing) ##########"
rocm-smi --showclocks 2>&1 | grep sclk | head -8

status=0
for dir in 0 1; do
  name="${TAG}_$([ "$dir" = 0 ] && echo fwd || echo rev)"
  echo
  echo "########## pass $name (reverse=$dir) ##########"
  wait_clean || { echo "AB REFUSED: node never went clean"; exit 1; }
  docker exec -w $ON/harness -e AB_TAG="$name" dhk-gemmrs timeout 5400 \
    python3 -u "$D/ab_pershape.py" "$ROUNDS" "$ITERS" 600 "$dir" 2>&1 \
    | tee "$D/logs/ab_${name}.log"
  rc=${PIPESTATUS[0]}
  echo "pass $name exit status: $rc"
  [ "$rc" = "0" ] || status=$rc
done

echo
echo "AB DONE (status $status); logs in $D/logs"
exit $status
