#!/usr/bin/env bash
# exp_35 step 2: what does K0_MPS_TRACE / DESC_DUMP actually print, and how do I
# drive a campaign + read arm_p50_us. Also: which env var turns on timestamps.
set -uo pipefail
AB=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py
BM=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill

echo "===TRACE_PRINT_SITES==="
for L in 3389 4560; do echo "--- around $L ---"; sed -n "$((L-14)),$((L+16))p" "$AB"; done

echo
echo "===CFG_PARSE_IN_HOST==="
grep -nE 'K0_MPS_CFG|encode_config|group_slices|flush_rows|timestamps|K0_MPS_TIMESTAMPS|DESC_DUMP' "$AB" | head -60

echo
echo "===SCREEN_SH==="
cat "$HOME/tools/screen.sh"

echo
echo "===RUN_CAMPAIGN_HEAD==="
sed -n '1,80p' "$BM/run_campaign.sh"
# end
