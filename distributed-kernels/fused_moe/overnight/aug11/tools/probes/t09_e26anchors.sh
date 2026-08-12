#!/usr/bin/env bash
# t09: READ-ONLY. exp_26 activate.md §4 anchors + their uniqueness.
set -uo pipefail
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
echo "== occurrences of each anchor string =="
for s in 'n2_phase1_gm.cpp' 'n2_phase1_gm_mps.cpp' 'n2_phase2_gm_mps.cpp' \
         'PF6_N2_FILES' 'PF6MPS_SOURCE_DIR' 'PF6_N2_SOURCE_DIR' 'PF6MPS_REQUESTED'; do
  printf '%-24s %s\n' "$s" "$(grep -c -- "$s" "$AB")"
done
echo
echo "== 4a: PF6_N2_FILES block =="
grep -n 'PF6_N2_FILES' "$AB" | head
sed -n "$(grep -n 'PF6_N2_FILES *=' "$AB" | head -1 | cut -d: -f1),+22p" "$AB"
echo
echo "== 4c: G-stack source contract (every n2_phase1_gm.cpp site) =="
grep -n 'n2_phase1_gm.cpp' "$AB"
echo "-- context of each --"
for L in $(grep -n 'n2_phase1_gm.cpp' "$AB" | cut -d: -f1); do
  echo "---- line $L ----"; sed -n "$((L-8)),$((L+6))p" "$AB"
done
echo
echo "== 4b: source-dir selector =="
grep -n '_source_name == "n2_phase2_gm_mps.cpp"' "$AB"
for L in $(grep -n '_source_name == "n2_phase2_gm_mps.cpp"' "$AB" | cut -d: -f1); do
  echo "---- line $L ----"; sed -n "$((L-12)),$((L+8))p" "$AB"
done
echo "== done =="
exit 0
