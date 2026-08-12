#!/usr/bin/env bash
# exp_27 pre-flight, READ-ONLY. Everything the design says must be true before
# a single byte is edited:
#   (1) node idle + node checkout == our HEAD
#   (2) nvi[1] really is T_ext  (the group-major stride M6 uses today)
#   (3) sc_stage row stride really is 56 floats (K0P6_NG)
#   (4) sc_stage base is >=16 B aligned and covers T_ext rows  (float4 load)
#   (5) nothing but M6 reads sc_dst
set -uo pipefail
DHK="$HOME/Distributed-HipKittens"
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
FM="$DHK/distributed-kernels/fused_moe"

echo "===== (0) node state ====="
hostname; date -u
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
git -C "$DHK" fetch --all -q
git -C "$DHK" log --oneline -1
echo "node HEAD  : $(git -C "$DHK" rev-parse HEAD)"
echo "origin HEAD: $(git -C "$DHK" rev-parse origin/codex/distributed-hipkittens-scaffold)"
git -C "$DHK" status --short | head

echo
echo "===== (1) where do hkp headers live ====="
for f in hkp_quant.hpp hkp_sort.hpp; do
  find "$HOME/amd-master" "$DHK" -name "$f" 2>/dev/null | head -3
done

echo
echo "===== (2) k0p6_sort::scan -- what lands in nvi[1] ====="
SORT="$(find "$HOME/amd-master" "$DHK" -name 'hkp_sort.hpp' 2>/dev/null | head -1)"
echo "file: $SORT"
if [ -n "$SORT" ]; then
  awk '/void scan\(/,/^\}/' "$SORT" | head -60
fi

echo
echo "===== (3) k0p6_unpack_row_nopoll -- sc_stage row stride ====="
grep -rn 'k0p6_unpack_row_nopoll' "$HOME/amd-master" "$DHK" --include=*.hpp --include=*.cuh --include=*.h 2>/dev/null | head -5
UNP="$(grep -rln 'k0p6_unpack_row_nopoll' "$HOME/amd-master" "$DHK" --include=*.hpp --include=*.cuh --include=*.h 2>/dev/null | head -1)"
echo "file: $UNP"
if [ -n "$UNP" ]; then
  awk '/k0p6_unpack_row_nopoll/,/^\}/' "$UNP" | head -70
fi

echo
echo "===== (4) hkp_quant.hpp zero_part_scale_transpose (the fused donor) ====="
QNT="$(find "$HOME/amd-master" "$DHK" -name 'hkp_quant.hpp' 2>/dev/null | head -1)"
if [ -n "$QNT" ]; then
  awk '/zero_part_scale_transpose/,/^\}/' "$QNT" | head -30
fi

echo
echo "===== (5) host allocation of SC_STAGE (slot 9) and SC_DST (slot 20) ====="
grep -rn 'sc_stage\|SC_STAGE' "$K0/prefill_opt/host/e004pf_k0pf_ab.py" 2>/dev/null | head -20

echo
echo "===== (6) every sc_dst reference in the MPS arm ====="
grep -n 'sc_dst\|SC_DST' "$FM/k0pf6gm_device_tile_mps.hip" "$FM/moe_mps_adapter.cuh" \
    "$FM/n2_phase1_gm_mps.cpp" "$FM/n2_phase2_gm_mps.cpp" 2>/dev/null

echo
echo "===== (7) current knobs ====="
grep -nE '^#define K0P6_MPS_SRC_REV|^#define N2GM_P1_SCHED_GSCALE|^#include "n2_phase1_gm(_mps)?\.cpp"' \
    "$FM/k0pf6gm_device_tile_mps.hip"
echo "done"
