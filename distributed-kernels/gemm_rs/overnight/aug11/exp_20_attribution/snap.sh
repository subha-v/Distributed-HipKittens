#!/usr/bin/env bash
# exp_20: read-only node snapshot -- job status, KFD occupancy, clocks.
# Clocks are captured into the artifact tree at every snapshot so the timing run
# can be shown to have happened at a pinned, determinism-level clock rather than
# asserted to have.
#   snap.sh <tag> [tail_lines]
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
TAG=${1:-snap}
N=${2:-30}
mkdir -p "$OUT/clocks"

if [ -f "$OUT/.abl.pid" ] && kill -0 "$(cat "$OUT/.abl.pid")" 2>/dev/null; then
  echo "STATUS: running pid=$(cat "$OUT/.abl.pid")"
else
  echo "STATUS: not running (finished or never started)"
fi
echo "KFD pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"

C=$OUT/clocks/$TAG.txt
{ date -u +%Y-%m-%dT%H:%M:%SZ; rocm-smi --showclocks; rocm-smi --showperflevel; } \
  > "$C" 2>&1
echo "--- clocks/$TAG.txt (sclk + perf level, one line per GPU) ---"
grep -E 'sclk|Performance Level' "$C" | head -20

echo "--- tail -$N reattribute_run.log ---"
tail -n "$N" "$OUT/reattribute_run.log" 2>&1
