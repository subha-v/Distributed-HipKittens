#!/usr/bin/env bash
K=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
AB=$K/prefill_opt/host/e004pf_k0pf_ab.py
echo "===_parse_mps_config (lines 370-440)==="
sed -n '370,440p' $AB
echo; echo "===print sites for our tags==="
grep -n 'MPS TS\|MPS SOAK\|MPS SPIN\|MOK GATE\|\[MARK\]\|MOK SYNTHETIC EAGER\|K0PF PROFILE\|K0PF GATE' $AB | head -40
echo; echo "===BLOCKED_STATUSES body==="
sed -n '20,45p' $K/benchmarks/mok_synthetic_prefill/summarize.py
exit 0
