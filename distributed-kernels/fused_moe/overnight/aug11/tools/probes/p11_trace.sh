#!/usr/bin/env bash
K=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
AB=$K/prefill_opt/host/e004pf_k0pf_ab.py
echo "===K0_MPS_TRACE sites==="; grep -n 'K0_MPS_TRACE' $AB
echo; echo "===context==="; grep -n -A6 -B2 'K0_MPS_TRACE' $AB | head -40
echo; echo "===lock dir==="; ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1
echo; echo "===disk==="; df -h $HOME | tail -2
echo; echo "===rocm-smi showpids==="; /opt/rocm/bin/rocm-smi --showpids 2>&1 | tail -20
echo; echo "===JIT dir==="; ls -t $HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5/*/k0pf6gm_mps_mega.hsaco 2>/dev/null | head -3 | xargs -r stat -c '%y %n'
echo; echo "===SRC_REV==="; grep -oE 'K0P6_MPS_SRC_REV [0-9]+' $HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip | tail -1
exit 0
