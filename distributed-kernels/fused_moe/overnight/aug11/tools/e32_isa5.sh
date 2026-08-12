#!/usr/bin/env bash
# exp_32 Job 2 step 5: dump the mode-12 M8 instantiation (the ONLY one with 20
# flat_store_dwordx4 per c-iteration = 16 guarded zero stores + 4 out stores).
# READ-ONLY.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
ISA=$SC/out2/D.isa

echo "===A. exec-mask / branch structure of the whole M8-mode12 body (24460..25130)==="
awk 'NR>=24460 && NR<=25130' "$ISA" | sed 's|//.*||' | sed 's/[[:space:]]*$//' \
  | awk 'NF{printf "%d\t%s\n", NR+24459, $0}' \
  | grep -E 'ds_bpermute|flat_load_dwordx4|flat_store_dwordx4|s_and_saveexec|s_andn2_saveexec|s_cbranch|s_or_b64 *exec|s_mov_b64 *exec|v_cmp_lt_i32|s_waitcnt|s_branch|BB|^[0-9]+\t;'

echo
echo "===B. VERBATIM 24495..24625 (the load block)==="
awk 'NR>=24495 && NR<=24625 {printf "%d\t%s\n", NR, $0}' "$ISA"
exit 0
