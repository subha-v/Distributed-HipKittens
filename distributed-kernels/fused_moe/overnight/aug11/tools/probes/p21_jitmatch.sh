#!/usr/bin/env bash
J=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
TARGET=$(md5sum $J/latest/k0pf6gm_mps_mega.hsaco | cut -d' ' -f1)
echo "latest md5: $TARGET"
echo "=== searching hashed dirs for a content match ==="
for f in $(ls -t $J/*/k0pf6gm_mps_mega.hsaco | grep -v '/latest/' | head -40); do
  m=$(md5sum "$f" | cut -d' ' -f1)
  if [ "$m" = "$TARGET" ]; then echo "MATCH $(basename $(dirname $f))  $(stat -c %y $f | cut -c1-19)"; fi
done
echo "(end of match scan)"
echo
echo "=== what did the verify log say around its jit path? ==="
grep -n 'jit/gfx950_mlx5' $HOME/overnight-scratch/verify_C64g1mode2flush_rows16.log | head -5
echo
echo "=== does the new a11base log mention mori jit at all? ==="
L=$(ls -t $HOME/overnight-scratch/a11base_1_*.log | head -1)
grep -nic 'jit' "$L"
grep -ni 'mori' "$L" | head -5
echo
echo "=== latest/ dir contents ==="
ls -la $J/latest | head -20
exit 0
