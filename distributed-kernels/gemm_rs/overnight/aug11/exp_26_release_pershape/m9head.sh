#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
echo "########## m9 log, first case in full ##########"
sed -n '1,40p' "$D/logs/m9.log"
echo
echo "########## golden module provenance ##########"
ls -la "$ON/harness/build/gemm_rs_mi300x_e3base.so" "$ON/harness/build/gemm_rs_mi300x.so"
echo "-- how the golden was built --"
grep -nE 'RELEASE_GROUP|e3base|hipcc' "$ON/experiments/exp_05_release_granularity/build_golden.sh" 2>/dev/null | head -20
echo
echo "########## has m9 EVER passed on this node? (prior logs) ##########"
grep -rlE 'GATE M9 (PASSED|FAILED)' "$ON" --include='*.log' 2>/dev/null | head -10
for f in $(grep -rlE 'GATE M9 (PASSED|FAILED)' "$ON" --include='*.log' 2>/dev/null | head -6); do
  echo "-- $f  ($(stat -c%y "$f" | cut -c1-19))"
  grep -E 'GATE M9 (PASSED|FAILED)|golden itself' "$f" | tail -4
done
echo
echo "########## stuck proc ##########"
for p in $(pgrep -f 'm9_stale_slot'); do
  echo "pid $p state=$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null) wchan=$(cat /proc/$p/wchan 2>/dev/null)"
done
pgrep -f 'm9_stale_slot' >/dev/null || echo "  (gone)"
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -3
