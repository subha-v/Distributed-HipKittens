#!/usr/bin/env bash
# exp_33 probe 3: is there ANY reachable per-phase signal for pf6gm_mega (for the
# M7 epilogue surcharge) and for production, through the campaign container?
# Read-only.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
RC="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"

echo "=== K0_PF6GM_DECOMP block (ab.py 5219..5310) ==="
sed -n '5219,5312p' "$AB"

echo
echo "=== run_campaign.sh full -e forwarding list ==="
grep -nE '^\s*-e ' "$RC"

echo
echo "=== production stage names _prs ==="
grep -n '_prs' "$AB" | sed -n '1,10p'

echo
echo "=== driver progress so far ==="
tail -n 25 "$HOME/e33/e33a_driver.log" 2>/dev/null || echo "(no driver log yet)"
echo "--- campaign log tail ---"
LG=$(ls -t $HOME/overnight-scratch/e33a_*.log 2>/dev/null | head -1)
echo "log=$LG"
[ -n "$LG" ] && tail -n 12 "$LG"
echo "=== PROBE3 DONE ==="
exit 0
