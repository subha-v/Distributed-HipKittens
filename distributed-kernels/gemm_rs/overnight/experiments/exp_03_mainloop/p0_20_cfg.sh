#!/usr/bin/env bash
# P0 step 2: exact CFG of the <256,256,32,false> region around the k-loop, plus
# the raw text at every scratch site. READ-ONLY.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s
L=$EXP/logs

echo "############ metadata block, raw, for the two 256/256/32 symbols ############"
grep -n -E '\.agpr_count|\.max_flat_workgroup_size|\.name:|\.private_segment_fixed_size|\.sgpr_count|\.sgpr_spill_count|\.symbol:|\.vgpr_count|\.vgpr_spill_count|\.group_segment_fixed_size|\.occupancy|\.uses_dynamic_stack|\.workgroup_processor_mode' "$S" \
  | sed 's/^\([0-9]*\):\s*/\1  /' > "$L/p0_meta_raw.txt"
wc -l "$L/p0_meta_raw.txt"
cat "$L/p0_meta_raw.txt"

echo
echo "############ CFG: all labels and branches in lines 9208..14172 (<256,256,32,false>) ############"
awk 'NR>=9208 && NR<=14172' "$S" | grep -n -E '^\.LBB|s_cbranch|s_branch|s_setpc|s_endpgm|; %bb\.' \
  | awk -F: -v base=9207 '{printf "%d  %s\n", $1+base, substr($0, index($0,":")+1)}' > "$L/p0_cfg_256_256_32_false.txt"
wc -l "$L/p0_cfg_256_256_32_false.txt"
echo "--- CFG restricted to the k-loop neighbourhood 10480..11480 ---"
awk '$1>=10480 && $1<=11480' "$L/p0_cfg_256_256_32_false.txt"

echo
echo "############ every scratch site with 8 lines of context ############"
for ln in 10287 10514 10622 11401 15298 15553 15672 16586; do
  echo "-------- scratch site near line $ln --------"
  awk -v a=$((ln-8)) -v b=$((ln+8)) 'NR>=a && NR<=b {printf "%s %7d| %s\n", (NR==ARGV[2]?">>":"  "), NR, $0}' "$S" "$ln" 2>/dev/null \
    || awk -v a=$((ln-8)) -v b=$((ln+8)) -v t=$ln 'NR>=a && NR<=b {printf "%s %7d| %s\n", (NR==t?">>":"  "), NR, $0}' "$S"
done
echo "=============== DONE ==============="