#!/usr/bin/env bash
# Independent repeat of M7 only, against whatever is currently built.
#   remeasure.sh <arm_name> <tag>
# Same command, same rotations, same iteration counts as the ladder's M7. Used
# to confirm a delta rather than to replace the ladder.
set -uo pipefail
ARM=${1:?usage: remeasure.sh <arm> <tag>}
TAG=${2:-confirm}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$ON/experiments/exp_03_mainloop/arms/$ARM
mkdir -p "$L"

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || { echo "node dirty, refusing to time"; exit 1; }

docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u m7_bench.py 3 50 2>&1 | tee "$L/m7_bench_$TAG.log"
cp $ON/harness/m7_results.json "$L/m7_results_$TAG.json" 2>/dev/null || true
echo
sed -n '/GATE M7/,$p' "$L/m7_bench_$TAG.log"
