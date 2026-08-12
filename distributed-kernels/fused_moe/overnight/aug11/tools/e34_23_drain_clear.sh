#!/usr/bin/env bash
# exp_34 condition 2 (concurrent clear) + the drain-confound source read.
set -uo pipefail
D=$HOME/e34/DHK/distributed-kernels/fused_moe
KRN=$D/k0pf6gm_device_tile_mps.hip
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
HOST=$K0/prefill_opt/host/e004pf_k0pf_ab.py

echo "################ 2a. mori_t (does it zero?) ################"
grep -n -A14 "^def mori_t" "$HOST"
echo
echo "################ 2b. row_ready alloc site, in context ################"
sed -n '1495,1510p' "$HOST"
echo
echo "################ 2c. use at 2715-2725 ################"
sed -n '2712,2726p' "$HOST"
echo
echo "################ 2d. use at 5225-5245 (possible clear) ################"
sed -n '5222,5248p' "$HOST"
echo
echo "################ 2e. is that inside the soak / epoch loop? ################"
awk 'NR>=5150 && NR<=5245 && (/for .*range/ || /while /|| /def /|| /SOAK|soak/)' "$HOST" | head -20
echo "--- named function containing 5235 ---"
awk 'NR<=5235 && /^def |^    def /{l=NR": "$0} END{}' "$HOST" >/dev/null
awk 'NR<=5235 && /^def /{ln=NR; s=$0} END{print ln": "s}' "$HOST"
echo
echo "################ DRAIN HOOKS: KRN 320-420 verbatim ################"
sed -n '320,420p' "$KRN"
echo "===DONE==="
