#!/usr/bin/env bash
# Close-out: what is on the node, built from what, and the graded geomeans as the
# report tool printed them (rather than as anyone recomputed them).
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
GEMM=/home/subvadla/dhk/distributed-kernels/gemm_rs
D=$ON/experiments/exp_05_release_granularity

echo "===== shipped source ====="
sha256sum $GEMM/gemm_rs_mi300x.cpp
grep -n 'define HK_GEMM_RS_MI300X_RELEASE_GROUP' $GEMM/gemm_rs_mi300x.cpp
echo "pre-E3 golden:"
sha256sum $D/baseline/gemm_rs_mi300x_e3base.cpp

echo
echo "===== built modules ====="
ls -l --time-style=+%H:%M:%S $ON/harness/build/*.so | awk '{print $6, $7, $8}'

echo
echo "===== archived arms ====="
for a in N1 N2 N4 N4c; do
  printf '%-5s ' "$a"
  ls $D/arms/$a 2>/dev/null | tr '\n' ' '
  echo
done

echo
echo "===== graded geomeans, verbatim from vs_report.py ====="
for t in rg1 rg4c rg4c_r2; do
  echo "-- $t --"
  python3 $ON/experiments/exp_10_rank1/vs_report.py $D/vs_logs/$t 2>/dev/null \
    | sed -n '/GEOMETRIC MEANS/,/^====/p' | grep -E 'arm|ours|rank-1|geo '
done

echo
echo "===== graded shape 6, all three runs ====="
for t in rg1 rg4c rg4c_r2; do
  printf '%-9s ' "$t"
  python3 $ON/experiments/exp_10_rank1/vs_report.py $D/vs_logs/$t 2>/dev/null \
    | grep '8192x8192x29568' | awk '{printf "%s %s  ", $3, $4}'
  echo
done

echo
echo "===== node clean ====="
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -3
echo "===== DONE ====="
