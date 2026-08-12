#!/usr/bin/env bash
# aug11/exp_20: launch the attribution refresh DETACHED.
#
# Detached because the run is ~30-60 min of GPU time and an ssh hiccup must not
# kill it. tools/reattribute.sh is used verbatim (not reimplemented) because it
# carries the two guards this measurement needs: it force-removes every arm .so
# and the generated scratch first (exp_ablation.py otherwise prints "already
# built" and reports the PREVIOUS kernel's attribution), and it asserts shape 6's
# `full` against an expected total so a stale table cannot pass as a fresh one.
#
#   run_ablation.sh <expected_shape6_full_us> <tol_pct>
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution
EXPECT=${1:-1670}
TOL=${2:-6}
mkdir -p "$OUT"

if [ -f "$OUT/.abl.pid" ] && kill -0 "$(cat "$OUT/.abl.pid")" 2>/dev/null; then
  echo "ALREADY RUNNING pid=$(cat "$OUT/.abl.pid")"; exit 0
fi

# Node must be ours alone before a timing run.
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids before launch: $n"

echo "expect shape6 full ~= $EXPECT us +/- ${TOL}%"
rm -f "$OUT/reattribute_run.log"
setsid nohup bash "$ON/tools/reattribute.sh" "$EXPECT" "$TOL" \
  > "$OUT/reattribute_run.log" 2>&1 &
echo $! > "$OUT/.abl.pid"
sleep 5
echo "launched pid=$(cat "$OUT/.abl.pid")"
sed -n '1,20p' "$OUT/reattribute_run.log"
