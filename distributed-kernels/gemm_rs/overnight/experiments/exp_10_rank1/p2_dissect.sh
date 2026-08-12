#!/usr/bin/env bash
# exp_10 probe 2 -- NO GPU. Dissect rank-1's symmetric-heap bootstrap so we can
# drive ONE helper directly instead of through the evaluator, and read the
# actual exception from the previous attempt's 2.1 MB stderr.
set -uo pipefail

RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
OLD=/home/subvadla/dhk/.node/compbench/rank1

echo "===== A. shape of the submission ====="
wc -l "$RANK1"
grep -nE '^(def|class|CREATE_SHEMEM_CODE|[A-Z_]+ *=)' "$RANK1" | head -60

echo
echo "===== B. every mention of iris / ipc / heap_bases / subprocess / os.system ====="
grep -nE 'iris|ipc|Ipc|IPC|heap_bases|subprocess|os\.system|Popen|sudo|sleep|pickle' "$RANK1"

echo
echo "===== C. CREATE_SHEMEM_CODE verbatim ====="
awk '/CREATE_SHEMEM_CODE/{f=1} f{print NR": "$0} f&&/^"""|^\x27\x27\x27/{c++; if(c>=2) exit}' "$RANK1" | head -120

echo
echo "===== D. previous attempt: the actual exception in warm.stderr.txt ====="
if [ -f "$OLD/warm.stderr.txt" ]; then
  wc -l "$OLD/warm.stderr.txt"
  echo "--- unique Error/Exception lines with counts ---"
  grep -hoE '[A-Za-z_.]*(Error|Exception|Timeout)[A-Za-z_]*' "$OLD/warm.stderr.txt" | sort | uniq -c | sort -rn | head -20
  echo "--- first 60 lines ---"
  head -60 "$OLD/warm.stderr.txt"
  echo "--- first traceback block ---"
  grep -n -m1 -A 40 'Traceback' "$OLD/warm.stderr.txt"
else
  echo "absent"
fi

echo
echo "===== E. previous attempt: warm.stdout.txt (small, whole thing) ====="
cat "$OLD/warm.stdout.txt" 2>/dev/null | head -80

echo
echo "===== F. what wrote ipc_handles_rank*.bin ? ====="
grep -nE 'ipc_handles' "$RANK1" || echo "NOT rank-1's submission -> those files came from something else"
echo "===== DONE p2 ====="
