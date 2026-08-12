#!/usr/bin/env bash
# exp_10 probe 3 -- NO GPU. Two questions:
#   (1) Is CREATE_SHEMEM_CODE (and therefore heap_bases_*.pkl) ever actually
#       spawned, or is it dead scaffolding? The progress probe we have been
#       gating on may be measuring nothing.
#   (2) What exactly is at the fault site, submission.py:1643 and :1664 in
#       grouped_sum, where all 8 ranks took a read-only-page write fault?
set -uo pipefail

RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== A. is CREATE_SHEMEM_CODE ever used? ====="
grep -n 'CREATE_SHEMEM_CODE' "$RANK1"
echo "--- any process spawn at all? ---"
grep -nE 'subprocess|Popen|multiprocessing|os\.spawn|os\.fork|os\.exec|os\.system' "$RANK1"

echo
echo "===== B. load_heap_base_ptr + close_heap_base_ptr (1545..1615) ====="
sed -n '1545,1615p' "$RANK1" | cat -n | awk '{printf "%d: %s\n", $1+1544, substr($0, index($0,$2))}'

echo
echo "===== C. grouped_sum and the reduce kernel (1616..1690) ====="
awk 'NR>=1616 && NR<=1690 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== D. launch_triton_kernel around the call into grouped_sum (1436..1552) ====="
awk 'NR>=1500 && NR<=1552 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== E. line map: staged copy is +3 lines from the patch. Confirm. ====="
OLD=/home/subvadla/dhk/.node/compbench/rank1/submission.py
if [ -f "$OLD" ]; then
  echo "--- staged copy lines 1638..1670 (the traceback line numbers) ---"
  awk 'NR>=1638 && NR<=1670 {printf "%d: %s\n", NR, $0}' "$OLD"
fi
echo "===== DONE p3 ====="
