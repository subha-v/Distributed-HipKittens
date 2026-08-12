#!/usr/bin/env bash
K=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
AB=$K/prefill_opt/host/e004pf_k0pf_ab.py
echo "===4575-4600 MOK GATE==="; sed -n '4575,4600p' $AB
echo; echo "===4885,4905 EAGER==="; sed -n '4885,4905p' $AB
echo; echo "===4965,5032 SOAK/TS/SPIN==="; sed -n '4965,5032p' $AB
echo; echo "########## MPS_OVERNIGHT_HARNESS_NOTE.md ##########"
cat $K/benchmarks/mok_synthetic_prefill/MPS_OVERNIGHT_HARNESS_NOTE.md
exit 0
