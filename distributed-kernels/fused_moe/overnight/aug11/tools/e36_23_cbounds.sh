#!/usr/bin/env bash
# exp_36: what is the legal range of C (service-pool CTAs) in mode 12?
# The campaign ordering says smaller C is better, so the question is how far
# down the placement axis can legally be driven without mode 14.
set -u
F="$HOME/Distributed-HipKittens/distributed-kernels/fused_moe"
grep -n 'service\|nservice\|C <\|C >\|c\.C\|comm_ctas\|kMinC\|kMaxC' "$F/moe_mps_adapter.cuh" | head -40
echo "=== validate() in the adapter ==="
grep -n 'validate\|invalid\|reject' -A 25 "$F/moe_mps_adapter.cuh" | sed -n '1,80p'
exit 0
