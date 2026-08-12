#!/usr/bin/env bash
# exp_24 probe 2: read hkp::zero_part_scale_transpose verbatim + the -I roots
# the campaign build uses (so I know where a sibling helper can legally live).
set -uo pipefail
Q="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/solution/hip/hkp/hkp_quant.hpp"
echo "=== $Q  (lines 100-185)"
sed -n '100,185p' "$Q"
echo
echo "=== sha256 of hkp_quant.hpp"
sha256sum "$Q"
echo
echo "=== is hkp_quant.hpp inside the mori jit-sources tree?"
ls -l /usr/local/lib/python3.12/dist-packages/mori/_jit-sources 2>/dev/null | head -5
echo "(host has no mori; that path is container-only)"
echo
echo "=== how the campaign compiles the mps kernel: grep include dirs"
grep -rn "Distributed-HipKittens\|include_directories\|target_include\|_jit-sources\|hkp" \
  "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill/run_campaign.sh" | head -40
exit 0
