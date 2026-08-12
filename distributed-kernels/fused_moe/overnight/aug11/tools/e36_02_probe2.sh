#!/usr/bin/env bash
# exp_36 step 2: the mok_synthetic input generator -- where do tokens and
# routing skew come from in THIS input mode (K0_SYNTH_ROUTE is refused here)?
set -u
K0="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
AB="$K0/prefill_opt/host/e004pf_k0pf_ab.py"
SI="$(find "$K0" -name 'synthetic_inputs.py' | head -1)"
echo "SI = $SI"
echo "=== MoKSyntheticConfig ==="
sed -n '1,140p' "$SI"
echo
echo "=== ab.py 1855-1960 (mok config build) ==="
sed -n '1855,1960p' "$AB"
exit 0
