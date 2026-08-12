#!/usr/bin/env bash
date -u +%FT%TZ
echo "=== KFD processes ==="
rocm-smi --showpids 2>/dev/null | sed -n '1,40p'
echo "=== our job? ==="
pgrep -af 'torch.distributed.run|run_campaign|torchrun|mpirun' | head -20 || echo "(none)"
echo "=== load ==="
uptime
echo "=== newest hsaco ==="
find "$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5" -name '*.hsaco' -printf '%T@ %TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort -rn | head -5
