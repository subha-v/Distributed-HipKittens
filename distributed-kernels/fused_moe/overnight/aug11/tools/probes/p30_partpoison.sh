#!/usr/bin/env bash
# exp_24 probe 4: the exact host-side `part` lifecycle around the mps_mega arm.
set -uo pipefail
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
echo "=== ab.py 1698-1715 (the 'part' list)"
sed -n '1698,1716p' "$AB"
echo
echo "=== ab.py 5925,5950 (part fill 1000)"
sed -n '5925,5950p' "$AB"
echo
echo "=== ab.py 6120,6150"
sed -n '6120,6150p' "$AB"
echo
echo "=== ab.py 6660,6700"
sed -n '6660,6700p' "$AB"
echo
echo "=== all pf6mps / mps_mega body definitions and their part handling"
grep -n "pf6mps\|mps_mega\|def .*mps" "$AB" | head -60
exit 0
