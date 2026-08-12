#!/usr/bin/env bash
# exp_33 probe: how are [MPS TS*] / [MPS SPIN] emitted (rank-guarded? prefixed?),
# and what exactly does [K0PF PROFILE] production measure? Read-only.
set -uo pipefail
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
RC="$K0/benchmarks/mok_synthetic_prefill/run_campaign.sh"

echo "=== ab.py 5060..5125 (MPS TS emit block) ==="
sed -n '5055,5125p' "$AB"

echo
echo "=== ab.py stage_profile production block 4240..4275 ==="
sed -n '4240,4275p' "$AB"

echo
echo "=== ab.py 4052..4120 (stage profile construction) ==="
sed -n '4052,4120p' "$AB"

echo
echo "=== run_campaign.sh: redirects / per-rank logs / tee ==="
grep -nE 'redirect|tee|--log|log_dir|rank|torch.distributed.run|nproc' "$RC" | sed -n '1,40p'

echo
echo "=== does anything write stage_profile into the rank json? ==="
grep -n 'stage_profile' "$AB" | sed -n '1,40p'
echo "=== PROBE DONE ==="
exit 0
