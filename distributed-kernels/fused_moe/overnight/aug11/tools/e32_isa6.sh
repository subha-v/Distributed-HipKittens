#!/usr/bin/env bash
# exp_32 Job 2 step 6: exact census of the mode-12 M8 (c,jb) iteration body and the
# guard shape, plus proof that the 15 hoisted loads are consumed only inside guards.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
ISA=$SC/out2/D.isa

echo "===LOOP BOUNDS: backward s_cbranch targets near the M8-mode12 block==="
awk 'NR>=24400 && NR<=25200' "$ISA" | grep -nE 's_cbranch_(scc0|scc1|vccz|vccnz|execz|execnz) 6[0-9]{4}|s_branch 6[0-9]{4}' \
  | awk -F: '{printf "%d\t%s\n", $1+24399, $2}' | head -40

echo
echo "===CENSUS of 24494..25000 (one (c,jb) iteration of mode-12 M8)==="
for m in flat_load_dwordx4 flat_store_dwordx4 ds_bpermute_b32 v_lshl_add_u64 \
         v_cmp_lt_i32_e32 s_and_saveexec_b64 's_or_b64 exec' s_cbranch_execz \
         s_cbranch_execnz 's_waitcnt vmcnt(0)' 's_waitcnt lgkmcnt(0)' v_pk_add_f32 \
         v_lshlrev_b32_e32 s_nop; do
  printf '%-26s %s\n' "$m" "$(awk 'NR>=24494 && NR<=25000' "$ISA" | grep -c "$m")"
done
printf '%-26s %s\n' "TOTAL non-blank lines" "$(awk 'NR>=24494 && NR<=25000' "$ISA" | grep -cE '^\s*[a-z_]')"

echo
echo "===THE 16 GUARD COMPARES in the block (operand pairs)==="
awk 'NR>=24494 && NR<=25200' "$ISA" | grep -n 'v_cmp_lt_i32_e32' | sed 's|//.*||' \
  | awk -F: '{printf "%d\t%s\n", $1+24493, $2}'

echo
echo "===ds_bpermute OFFSETS in the unconditional prologue 24500..24572==="
awk 'NR>=24500 && NR<=24572' "$ISA" | grep 'ds_bpermute' | sed 's|//.*||' \
  | sed 's/^[[:space:]]*//' | awk '{print $NF, $0}' | sort -u | awk '{$1=""; print}'

echo
echo "===jb=4 HALF: is there a sibling block with offsets 16/20/24/28? ==="
grep -n 'ds_bpermute_b32 .*offset:2[048]$\|ds_bpermute_b32 .*offset:16$' "$ISA" | sed 's|//.*||' | head -30
exit 0
