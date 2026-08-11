#!/usr/bin/env bash
# Normalize line endings on the node WITHOUT running push.ps1.
#
# Needed because push.ps1 also scps the kernel sources, and when another
# experiment is mid-edit on the Windows side a push would ship a half-written
# kernel -- or silently revert a validated one. This script touches only text
# files already on the node and never copies anything.
#
# Symptom it fixes: a CRLF `run_ours_evaluator.sh` whose trailing if/else/fi is
# corrupted, so BOTH the one-shape and full-suite branches execute and the
# second overwrites the first's stderr log.
set -u
REPO=/home/subvadla/dhk
REL=distributed-kernels/gemm_rs

echo "===== files containing CR before ====="
grep -rlU $'\r' $REPO/$REL --include='*.sh' --include='*.py' 2>/dev/null \
  | grep -v '/build/' | grep -v '/compbench/' | head -40
echo "  (count)"
grep -rlU $'\r' $REPO/$REL --include='*.sh' --include='*.py' 2>/dev/null \
  | grep -v '/build/' | grep -v '/compbench/' | wc -l

echo
echo "===== normalizing ====="
find $REPO/$REL -type f -writable \( -name '*.sh' -o -name '*.py' \) \
  -not -path '*/build/*' -not -path '*/compbench/*' \
  -exec sed -i 's/\r$//' {} +
echo "done"

echo
echo "===== files containing CR after ====="
grep -rlU $'\r' $REPO/$REL --include='*.sh' --include='*.py' 2>/dev/null \
  | grep -v '/build/' | grep -v '/compbench/' | wc -l

echo
echo "===== syntax check the evaluator runner ====="
bash -n $REPO/$REL/overnight/tools/run_ours_evaluator.sh && echo "  run_ours_evaluator.sh parses OK" \
  || echo "  run_ours_evaluator.sh STILL BROKEN"

echo "===== tail of the runner (the if/else/fi that was corrupted) ====="
tail -20 $REPO/$REL/overnight/tools/run_ours_evaluator.sh
echo "===== DONE ====="
