#!/usr/bin/env bash
# exp_35 step 3: what proves the decode empirically? DESC_DUMP contents, whether
# run_campaign.sh forwards it, and the arm_p50_us schema in summarize.py.
set -uo pipefail
AB=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py
BM=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill

echo "===DESC_DUMP_SITE==="
sed -n '3230,3300p' "$AB"

echo
echo "===ENV_FORWARDED_TO_DOCKER==="
sed -n '110,175p' "$BM/run_campaign.sh"

echo
echo "===SUMMARIZE_ARM_P50==="
grep -nE 'arm_p50_us|median|values|rank_max' "$BM/summarize.py" | head -40

echo
echo "===HOST_CFG_ECHO==="
grep -nE 'print\(.*(MPS CFG|mps cfg|cfg=)|MPS CFG' "$AB" | head -20
grep -nE '\[MPS TS\]|\[MPS TS SPLIT\]|\[MPS TS DELTA\]|\[MPS SPIN\]' "$AB" | head -20

echo
echo "===VALIDATE_PATH==="
grep -nE 'config_is_valid|invalid mps config|K0P6_MPS_ERR|refus' "$AB" | head -20
grep -rn 'config_is_valid' $HOME/Distributed-HipKittens/distributed-kernels/fused_moe/*.hip $HOME/Distributed-HipKittens/distributed-kernels/fused_moe/*.cuh | head -20
# end
