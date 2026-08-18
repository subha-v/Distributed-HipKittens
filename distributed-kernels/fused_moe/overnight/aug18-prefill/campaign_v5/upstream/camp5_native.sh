#!/usr/bin/env bash
# CAMPAIGN 5 (triarm1): the three-arm decomposition the clean story needs —
#   native  = TRUE PRODUCTION: untouched vllm/vllm-openai-rocm:v0.25.1, no
#             PF4H env, no apply.py, no patch chain (heavy prefill eager,
#             generic <=512 graphs — what users actually deploy)
#   stock   = patched stock control: production MoE ops inside the M15/M23
#             integration (B4096 stock graph + rescue + runtime resident)
#   m15     = the megakernel inside the same integration
# Ratios: m15/stock = kernel substitution; stock/native = integration effect;
# m15/native = the production headline.  2 order-balanced 3-arm rotations.
# Queued at the end of the overnight chain (waits for campaign 4).
set -uo pipefail
while ! grep -q "CAMP4_RC=" /home/subvadla/eplb_campaign/camp4_20260818.log 2>/dev/null; do
  sleep 60
done
sleep 30
export RUN_TAG=triarm1
export DATE_TAG=20260818
export PACKET=/home/subvadla/amd-master-m15pkt/auto-gpu-kernel/k0_fused_moe/vllm_r1_aiter_e2e
export N2=/home/subvadla/pf4h_vllm_20260729/isolation_v1_20260729/n2/k0_n2.cpython-312-x86_64-linux-gnu.so
export M15_SOURCES=/home/subvadla/m20_deploy_sources_20260814
export DHK_ROOT=/home/subvadla/DHK-m20pkt
export CLIENT=/home/subvadla/pf4h_vllm_20260729/bench_exact_token_ids_v2.py
export GPU_CLAIM_NAME=m15pkt
export QSL_PKL=/home/subvadla/mlperf_v6_datasets/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
export PROMPT_SOURCE=qsl
export M23_PATCH=/home/subvadla/eplb_campaign/m23_patch.py
export EPLB_PREWARM=0
echo "=== CAMPAIGN 5 (triarm1): native vs stock vs m15 (c32p, 2 rotations) ==="
bash /home/subvadla/eplb_campaign/run_m15_campaign_eplb_v4.sh \
  --arms native,stock,m15 --cells c32p --pairs 2
echo "CAMP5_RC=$?"
