#!/usr/bin/env bash
# exp_35 step 1b: print the full g-field mechanism bit table and the throttle
# enable/depth predicates verbatim, so the rung-(b) g can be chosen to differ
# from g=353 in the throttle bits ONLY.
set -uo pipefail
ADP=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/moe_mps_adapter.cuh
echo "===ADAPTER_LINES_55_105==="
sed -n '55,105p' "$ADP"
echo
echo "===ADAPTER_LINES_215_300==="
sed -n '215,300p' "$ADP"
echo
echo "===ADAPTER_LINES_340_380==="
sed -n '340,380p' "$ADP"
echo
echo "===ALL_G_MASK_CONSTANTS==="
grep -nE 'inline constexpr std::uint32_t k[A-Za-z]+ *= *0x' "$ADP"
echo
echo "===ALL_G_PREDICATES==="
grep -nE '__host__ __device__ __forceinline__ (bool|std::uint32_t) [a-z_]+\(config' "$ADP"
# end
