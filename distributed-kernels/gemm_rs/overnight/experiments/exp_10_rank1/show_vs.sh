#!/usr/bin/env bash
# Show whatever the ours-vs-rank1 sweep produced, newest first. Read-only.
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1

echo "===== newest artifacts ====="
ls -lat $E/vs_logs/ 2>/dev/null | head -20

echo
echo "===== any summary table in the sweep log? ====="
for f in $(ls -t $E/vs_logs/*.txt $E/vs_logs/*.log 2>/dev/null | head -3); do
  echo "--- $f ---"
  tail -60 "$f"
done

echo
echo "===== node state ====="
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
echo "===== DONE ====="
