#!/usr/bin/env bash
# M7 gate timing at the same config, for the two columns the attribution cannot
# produce on its own: `bound` (is this shape's number describing the host?) and
# an independent same-config mean vector to cross-check the ablation's `full`.
#
# Run with cwd = THIS directory so m7_results.json lands here. harness_lib
# resolves build/ from its own __file__, not from cwd, so nothing under
# harness/ is read differently and nothing there is overwritten.
#
#   run_m7.sh [rotations] [iters]
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
ROT=${1:-3}
IT=${2:-50}
mkdir -p "$OUT"

wait_clean() {
  local n
  for ((i=0; i<60; i++)); do
    n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
    [ "$n" = "0" ] && { echo "node clean after ${i}0s"; return 0; }
    [ "$i" = "0" ] && echo "KFD pids: $n -- waiting for the node to drain"
    sleep 10
  done
  echo "ABORT: node still dirty"; return 1
}
wait_clean || exit 1

echo "===== clocks before ====="
rocm-smi --showperflevel 2>/dev/null | grep -c perf_determinism
rocm-smi --showclocks 2>/dev/null | grep -i sclk | head -2

echo "===== M7: $ROT rotations x $IT pipelined iters ====="
docker exec -w "$OUT" -e PYTHONPATH="$ON/harness" dhk-gemmrs timeout 2400 \
  python3 -u "$ON/harness/m7_bench.py" "$ROT" "$IT" 2>&1 | tail -30

echo "===== clocks after ====="
rocm-smi --showperflevel 2>/dev/null | grep -c perf_determinism
ls -la "$OUT/m7_results.json" 2>&1
echo "M7 DONE"
