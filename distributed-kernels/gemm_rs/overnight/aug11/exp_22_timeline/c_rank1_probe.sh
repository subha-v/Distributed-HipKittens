#!/usr/bin/env bash
# exp_22 arm (c): can the frozen rank-1 submission be traced NON-INVASIVELY?
#
# Read-only probe.  Launches nothing on the GPU; it only answers the three
# questions design.md section 8 says must hold before a traced pass is worth a
# node lease:
#
#   1. Is the frozen source still hash-clean, and does patch_rank1.py's
#      disclosed repair still apply to a COPY?  (The original is never edited.)
#   2. Is a warm staged arm already sitting in compbench/rank1, so the traced
#      pass can be the final `benchmark` and not a cold Triton compile?
#      run_rank1_bench3.sh's order is benchmark(warm) -> test -> benchmark, and
#      it exists because eval.py's test mode hardcodes a 60 s per-rank timeout.
#      DO NOT REORDER IT.
#   3. Does rocprofv3 exist in dhk-eval, and does that container's separate
#      filesystem already have a writable output path owned by uid 15523?
set -uo pipefail

NODE=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$NODE/compbench/rank1
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== 1. frozen source ====="
if [ -f "$RANK1" ]; then
  echo "present: $RANK1"
  sha256sum "$RANK1"
  echo "expected prefix: 7940fcb8 ... f0dc5"
else
  echo "MISSING -- arm (c) is not obtainable from this node"
fi

echo
echo "===== 2. staged arm state (is a warm Triton cache already there?) ====="
for d in "$DIR" "$DIR/.triton"; do
  if [ -d "$d" ]; then
    echo "$d: $(find "$d" -type f 2>/dev/null | wc -l) files, "\
"$(du -sh "$d" 2>/dev/null | cut -f1)"
  else
    echo "$d: absent"
  fi
done
ls -la "$DIR"/*.popcorn.txt 2>/dev/null || echo "no previous popcorn output"

echo
echo "===== 3. profiler availability in BOTH containers ====="
for c in dhk-gemmrs dhk-eval; do
  printf '%-12s ' "$c"
  docker exec "$c" /opt/rocm/bin/rocprofv3 --version 2>&1 | head -2 | tr '\n' ' '
  echo
done
echo "dhk-eval write test under the repo (must be chowned back to 15523):"
docker exec dhk-eval bash -c \
  "touch $NODE/aug11/exp_22_timeline/.evalprobe && ls -l $NODE/aug11/exp_22_timeline/.evalprobe && rm -f $NODE/aug11/exp_22_timeline/.evalprobe" \
  2>&1 | tail -2

echo
echo "===== VERDICT INPUTS COLLECTED ====="
echo "If all three are green, the traced pass is:"
echo "  rocprofv3 --kernel-trace wrapped around the FINAL benchmark pass of"
echo "  run_rank1_bench3.sh, rank 0 only, with the pass order untouched."
echo "If any is red, exp_22 ships TWO arms and result.md says exactly which"
echo "check failed. Adding profiler overhead to a competitor's number and then"
echo "reporting that number would be worse than shipping two arms."
