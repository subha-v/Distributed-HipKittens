#!/usr/bin/env bash
# Gate M9 against whatever is currently in harness/build. Refuses to start on a
# dirty node: it is an 8-GPU job like any other.
#
#   run_m9.sh [epoch_scale] [log_tag]
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity
SCALE=${1:-1}
TAG=${2:-run}
mkdir -p "$D/logs"

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || { echo "M9 REFUSED: node dirty (another GPU job is running)"; exit 1; }

docker exec -w $ON/harness dhk-gemmrs timeout 5400 \
  python3 -u m9_stale_slot.py "$SCALE" 2>&1 | tee "$D/logs/m9_${TAG}.log"
status=${PIPESTATUS[0]}
echo "m9 exit status: $status  (log: $D/logs/m9_${TAG}.log)"
exit $status
