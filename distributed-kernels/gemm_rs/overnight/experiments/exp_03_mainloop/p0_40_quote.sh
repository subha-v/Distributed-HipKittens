#!/usr/bin/env bash
# P0 step 4: quote the raw instruction text at the k-loop seams; prove there are
# no counted waitcnts anywhere. READ-ONLY.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s

dump() { awk -v a=$1 -v b=$2 'NR>=a && NR<=b {printf "%7d| %s\n", NR, $0}' "$S"; }

echo "############ every distinct s_waitcnt FORM in the whole .s, with counts ############"
grep -oE 's_waitcnt[^;]*' "$S" | sed 's/[[:space:]]*$//' | sort | uniq -c | sort -rn

echo
echo "############ any COUNTED (nonzero) wait anywhere? ############"
printf 'vmcnt(nonzero)   : %s\n' "$(grep -cE 'vmcnt\([1-9]' "$S")"
printf 'lgkmcnt(nonzero) : %s\n' "$(grep -cE 'lgkmcnt\([1-9]' "$S")"
printf 'expcnt(any)      : %s\n' "$(grep -cE 'expcnt\(' "$S")"

echo
echo "############ SEAM 1: tail of .LBB3_87 (ds_reads) -> guard -> head of %bb.88 ############"
dump 11135 11182

echo
echo "############ SEAM 2: tail of %bb.88 -> .LBB3_86 (MFMA block) head, exec order ############"
dump 11245 11262
echo "   ... control transfers to .LBB3_86 at 10957 ..."
dump 10955 10972

echo
echo "############ SEAM 3: end of MFMA block .LBB3_86 = latch/exit test ############"
dump 11018 11032

echo
echo "############ the 4 global_load_dwordx4 of one k-iteration, verbatim ############"
awk 'NR>=11153 && NR<=11260 && (/global_load|s_waitcnt|ds_write|ASMSTART|ASMEND/) {printf "%7d| %s\n", NR, $0}' "$S"

echo
echo "############ the 24 ds_read_b64 of one k-iteration, verbatim ############"
awk 'NR>=11029 && NR<=11152 && /ds_read/ {printf "%7d| %s\n", NR, $0}' "$S"

echo
echo "############ K_TAIL=true: the mask block %bb.89 head (16246..16262) ############"
dump 16246 16260

echo
echo "############ artifact listing ############"
ls -la "$EXP/logs" "$EXP/isa"
echo "=============== DONE ==============="