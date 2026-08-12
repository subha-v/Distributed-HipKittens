#!/usr/bin/env bash
L=$(ls -t $HOME/overnight-scratch/a11base_1_*.log | head -1)
echo "log: $L"
echo "=== all [mori-jit] lines (dedup) ==="
grep -h 'mori-jit' "$L" | sort | uniq -c | sort -rn | head -30
echo
echo "=== lines containing 'jit' that are not aiter ==="
grep -i 'jit' "$L" | grep -v '\[aiter\]' | sort -u | head -20
echo
echo "=== stat with and without -L on the symlink ==="
J=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
echo "no -L : $(stat -c %y $J/latest/k0pf6gm_mps_mega.hsaco)"
echo "with -L: $(stat -L -c %y $J/latest/k0pf6gm_mps_mega.hsaco)"
echo "readlink: $(readlink $J/latest/k0pf6gm_mps_mega.hsaco)"
exit 0
