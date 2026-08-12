#!/usr/bin/env bash
K=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
AB=$K/prefill_opt/host/e004pf_k0pf_ab.py
echo "===MPS_CFG parsing in ab.py==="
grep -n 'K0_MPS_CFG' $AB | head -20
echo; echo "===cfg key names (context around parse)==="
grep -n 'mps_cfg\|_MPS_KEYS\|allowed\|ValueError' $AB | grep -i 'mps\|cfg' | head -40
echo; echo "===ALL env vars referenced in ab.py==="
grep -oE '"K0_[A-Z0-9_]+"' $AB | sort -u
grep -oE "'K0_[A-Z0-9_]+'" $AB | sort -u
echo; echo "===other env (MORI/HSA/etc) in ab.py==="
grep -oE '"(MORI|HSA|HIP|TORCH|NCCL|RCCL)_[A-Z0-9_]+"' $AB | sort -u
echo; echo "===env vars in summarize.py==="
grep -n 'environ\|getenv' $K/benchmarks/mok_synthetic_prefill/summarize.py | head -20
echo; echo "===BLOCKED_STATUSES in summarize.py==="
grep -n 'BLOCKED_STATUSES\|status' $K/benchmarks/mok_synthetic_prefill/summarize.py | head -40
exit 0
