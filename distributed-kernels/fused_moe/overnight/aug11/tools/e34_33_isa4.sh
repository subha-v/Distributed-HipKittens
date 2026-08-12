#!/usr/bin/env bash
# exp_34 condition 3, close the loop: the branch target right after the mode-14
# acquire is the payload-load block. Show it.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUT=/home/subvadla/e34/out
echo "=== marker window, exact (10 before / 6 after BEGIN) ==="
L=$(grep -n "E34_M14_ACQ_BEGIN" $OUT/MK.s | cut -d: -f1)
sed -n "$((L-10)),$((L+8))p" $OUT/MK.s
echo
echo "=== the branch target .LBB0_3911 : first 26 instructions ==="
S=$(grep -n "^\.LBB0_3911:" $OUT/MK.s | cut -d: -f1)
echo "(label at asm line $S)"
sed -n "${S},$((S+26))p" $OUT/MK.s
echo
echo "=== is there ANY vector load between the acquire and that label? ==="
awk -v a=$L -v b=$S "NR>a && NR<b && /global_load|flat_load|buffer_load|ds_read/ {c++; print NR\": \"\$0} END {print \"count=\"c+0}" $OUT/MK.s
' 2>&1
echo "===DONE==="
