#!/usr/bin/env bash
# t03: READ-ONLY. mok_gate's contract (keys, collectivity) + eager-loop text.
set -uo pipefail
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
echo "== mok_gate 3470..3530 =="
sed -n '3470,3530p' "$AB"
echo "== _obuf =="
grep -n '_PROD_LIKE *=' "$AB" | head -3
sed -n "$(grep -n 'def _obuf' "$AB" | head -1 | cut -d: -f1),+12p" "$AB"
echo "== eager loop body 4500..4530 =="
sed -n '4500,4530p' "$AB"
echo "== correctness.py stats keys =="
grep -n 'stats\[\|"nonfinite"\|return {' "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/correctness.py" | head -30
echo "== done =="
exit 0
