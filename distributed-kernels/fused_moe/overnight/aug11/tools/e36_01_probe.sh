#!/usr/bin/env bash
# exp_36 step 1: find (a) the legal K0_SYNTH_ROUTE family names + their std,
# and (b) every knob that can move the token count T inside the container.
set -u
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"

echo "=== synthetic_routes.py location ==="
find "$K0" -name 'synthetic_route*' -o -name '*synthetic_routes*' | head
echo
SR="$(find "$K0" -name 'synthetic_routes.py' | head -1)"
if [ -n "$SR" ]; then
  echo "=== $SR (families + std) ==="
  grep -n 'FAMILIES\|def \|std\|hot\|uniform\|balanced' "$SR" | head -60
fi
echo
echo "=== ab.py: SYNTH_ROUTE handling ==="
grep -n 'SYNTH_ROUTE\|SYNTH_SEED\|synth_route' "$AB" | head -40
echo
echo "=== ab.py: token count / T derivation ==="
grep -n 'K0_T\b\|"K0_T"\|K0_MAXTOK\|K0_T_LOC_MAX\|num_tokens\|n_tokens\|tokens =\|T =\|K0_MOK_T' "$AB" | head -60
echo
echo "=== ab.py: mok_synthetic input build ==="
grep -n 'mok_synthetic' "$AB" | head -40
exit 0
