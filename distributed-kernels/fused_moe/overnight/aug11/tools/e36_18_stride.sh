#!/usr/bin/env bash
# exp_36: is the token capacity compiled into the megakernel family? If the
# per-source segment stride is a #define, T cannot be swept without a kernel
# edit -- which is outside this experiment's ownership.
set -u
F="$HOME/Distributed-HipKittens/distributed-kernels/fused_moe"
AB="$HOME/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
echo "=== K0P6_D_MAXTOK and friends (mps) ==="
grep -rn 'K0P6_D_MAXTOK\|K0P6C_NCHUNK_MAX\|MAXTOK\|D_MAXTOK' "$F"/*.hip "$F"/*.cuh 2>/dev/null | head -40
echo
echo "=== the same in the pf6gm (non-MPS) megakernel ==="
G="$(ls "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"/prefill_opt/kernels/*pf6gm*.hip 2>/dev/null | head -3)"
echo "$G"
for f in $G; do grep -n 'MAXTOK\|4096\|maxtok' "$f" | head -20; done
echo
echo "=== how the host tells the kernel about T / maxtok ==="
grep -n 'maxtok\|MAXTOK' "$AB" | head -40
exit 0
