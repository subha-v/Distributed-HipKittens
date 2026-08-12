#!/usr/bin/env bash
B=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/benchmarks/mok_synthetic_prefill
echo "===ls==="; ls -la $B
echo; echo "===wc==="; wc -l $B/run_campaign.sh $B/summarize.py $B/correctness.py $HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py 2>&1
echo; echo "########## run_campaign.sh ##########"; cat $B/run_campaign.sh
exit 0
