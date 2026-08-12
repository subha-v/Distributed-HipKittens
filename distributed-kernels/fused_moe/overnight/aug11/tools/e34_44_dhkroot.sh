#!/usr/bin/env bash
# exp_34: can the campaign be pointed at a PRIVATE tree? (needed to run the
# protocol negative control without editing the pinned checkout, which another
# agent now owns). CPU-only inspection.
set -uo pipefail
RC=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh
echo "### DHK_ROOT / WORKSPACE_ROOT definitions"
grep -n -E "DHK_ROOT|WORKSPACE_ROOT|GPU lease|lease|idle" "$RC" | head -30
echo
echo "### head of the script (env defaults)"
sed -n '1,60p' "$RC"
echo "===DONE==="
