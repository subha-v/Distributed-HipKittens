#!/usr/bin/env bash
# exp_34 conditions 1 + 2 + 5: the row_ready ALLOCATION SIZE (possible 4-byte
# heap overflow), any concurrent clear of it, and where bit 26 surfaces in logs.
# READ ONLY. NO GPU WORK.
set -uo pipefail
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe

echo "################ 1. row_ready ALLOCATION ################"
echo "--- every mention of row_ready / ROW_READY on the host side ---"
grep -rn -iE "row_ready" "$K0" \
  --include=*.py --include=*.cpp --include=*.hpp --include=*.cu --include=*.hip \
  2>/dev/null | grep -viE "^.*(device_tile|n2_phase)" | head -40
echo
echo "--- descriptor slot 25 assignment sites ---"
grep -rn -E "ROW_READY|\[25\]|slot 25|d\[25\]" "$K0" \
  --include=*.py --include=*.cpp --include=*.hpp 2>/dev/null | head -30
echo
echo "--- T_LOC_MAX / t_loc_max definition on the host ---"
grep -rn -iE "t_loc_max|T_LOC_MAX|TLOCMAX" "$K0" \
  --include=*.py --include=*.cpp --include=*.hpp 2>/dev/null | head -30

echo
echo "################ 2. CONCURRENT CLEAR ################"
echo "--- memset / zero_ / fill_ / .zero() near row_ready ---"
grep -rn -iE "row_ready.{0,60}(zero|memset|fill|\.zero_)|(zero_|memset|fill_).{0,60}row_ready" "$K0" \
  --include=*.py --include=*.cpp --include=*.hpp 2>/dev/null | head -20
echo "(empty above = no clear site found)"

echo
echo "################ 5. bit 26 / pperr in log output ################"
grep -rn -E "pperr|ERR_SERVICE|67108864" "$K0/prefill_opt/host/e004pf_k0pf_ab.py" 2>/dev/null | head -30
echo "===DONE==="
