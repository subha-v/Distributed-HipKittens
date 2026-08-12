#!/usr/bin/env bash
# exp_24 re-measure, PHASE 0c (retry): the geometry/rgroup assertion, as a real
# script file. The heredoc version reported rc=0 having printed nothing, which
# is what `python3 -` does when the heredoc never arrives -- a false pass.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=============== A. geometry + rgroup from the live shape planner ==============="
docker exec -w "$D" dhk-gemmrs python3 "$D/verify_rgroup.py" "$D/geometry.json"
rcA=$?
echo "  check A rc=$rcA"
[ $rcA -ne 0 ] && { echo "FATAL: rgroup did not reach the plan as expected"; exit 1; }

echo
echo "=============== D. rotation fix present on the node? ==============="
grep -n 'rot_mode\|rot_rng\|LAD_ROT\|arm_orders' "$D/ladder_mp.py" | sed 's/^/  /'
n=$(grep -c 'rot_mode' "$D/ladder_mp.py")
echo "  rot_mode occurrences: $n"
[ "$n" -lt 4 ] && { echo "FATAL: rotation fix is NOT on the node -- push failed"; exit 1; }
grep -n 'shift = rep % len(arms)' "$D/ladder_mp.py" | sed 's/^/  cyclic-path (kept for LAD_ROT=cyclic): /'
docker exec dhk-gemmrs python3 -m py_compile "$D/ladder_mp.py" && echo "  ladder_mp.py compiles"

echo
echo "=============== node state before the lease ==============="
if [ -f "$ON/tools/kfd_live.sh" ]; then
  . "$ON/tools/kfd_live.sh"
  echo "  live KFD holders: $(kfd_pids | tr '\n' ' ')"
else
  echo "  WARN tools/kfd_live.sh missing"
fi
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk|determinism' | head -4 | sed 's/^/  /'
echo "############ CHECKS COMPLETE ############"
