#!/usr/bin/env bash
J=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
echo "===is latest a symlink?==="; ls -ld $J/latest
echo "===count of hash dirs==="; ls -1 $J | wc -l
echo "===newest 5 by hsaco mtime==="; ls -t $J/*/k0pf6gm_mps_mega.hsaco 2>/dev/null | head -5 | xargs -r stat -c '%y %n'
echo "===python3 version==="; python3 -V
exit 0
