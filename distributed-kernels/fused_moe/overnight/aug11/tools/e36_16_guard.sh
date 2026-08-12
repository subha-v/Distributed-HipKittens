#!/usr/bin/env bash
# exp_36: the exact prefill-only T guard at ab.py:554 and anything else that
# pins T for the megakernel arms.
set -u
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
sed -n '520,575p' "$AB"
echo "=== other T pins ==="
grep -n 'prefill-only\|T != 4096\|T < \|T <= \|T >= \|T > ' "$AB" | head -30
echo "=== kernel-side T assumptions (mps) ==="
grep -rn '4096' "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | head -20
exit 0
