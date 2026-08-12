#!/usr/bin/env bash
# exp_36 step 3: (a) is there ANY skew knob reachable from mok_synthetic?
#                (b) full run_campaign.sh, so a scratch T-parameterised copy
#                    can be made without touching the harness.
#                (c) what T constraints does ab.py impose?
set -u
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
B="$K0/benchmarks/mok_synthetic_prefill"

echo "=== any skew/imbalance env knob in the mok_synthetic path? ==="
grep -rn 'skew\|SKEW\|imbalance\|hot_frac\|route_std' "$B" | head -20
echo "--- ab.py ---"
grep -n 'skew\|SKEW\|imbalance' "$AB" | head -20
echo
echo "=== synthetic_routes.py full head (skewed_hot definition) ==="
sed -n '1,80p' "$K0/prefill_opt/host/synthetic_routes.py"
echo
echo "=== T constraints in ab.py ==="
sed -n '110,145p' "$AB"
grep -n 'T %\|T must\|assert T\|padded\|PADMAX' "$AB" | head -20
echo
echo "=== run_campaign.sh (full) ==="
cat -n "$B/run_campaign.sh"
exit 0
