#!/usr/bin/env bash
# exp_27 pre-flight part 2: close the sc_stage index/alignment chain.
#   (a) k0p5_target -- does the unpack write sc_stage[row*56 + k]?
#   (b) T_LOC_MAX == T_ext == world*MAXTOK ?
#   (c) which tensor is descriptor slot 9 for the mps arm, and its alignment
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
CR="$(grep -rln 'k0p5_target' "$K0/solution/hip" 2>/dev/null | head -1)"

echo "===== (a) k0p5_target ====="
echo "file: $CR"
grep -rn 'k0p5_target' "$K0/solution/hip" 2>/dev/null | head
if [ -n "$CR" ]; then
  awk '/k0p5_target\(/,/^\}/' "$CR" | head -50
fi
echo
echo "-- K0P5 geometry constants --"
grep -rn 'K0P5_ROW_BYTES\|K0P5_PKT_DATA\|K0P5_PKTS\|K0P5_NG\|K0P5_ROW_STRIDE\|K0P5_SC_OFF\|K0P5_A_BYTES' \
    "$K0/solution/hip" 2>/dev/null | grep '#define' | head -20

echo
echo "===== (b) T_LOC_MAX / MAXTOK / NG in the host driver ====="
grep -nE '^ *(T_LOC_MAX|MAXTOK|NG|WORLD|K0P6_NG) *=' "$AB" | head -20
grep -n 'T_LOC_MAX *=' "$AB" | head -5

echo
echo "===== (c) descriptor slot 9 / 20 for the mps arm ====="
grep -n 'D_SC_STAGE\|D_SC_DST\|\[9\] *=\|\[20\] *=' "$AB" | head -30
echo "-- the pf6 descriptor build (look for sc_stage / sc_dst placement) --"
grep -n 'desc\[' "$AB" | grep -i 'sc_' | head -20

echo
echo "===== (d) pf6 state alloc for sc_stage/sc_dst (line 1740-1760, 2700-2710) ====="
sed -n '1740,1760p' "$AB"
echo "---"
sed -n '2695,2715p' "$AB"
echo done
