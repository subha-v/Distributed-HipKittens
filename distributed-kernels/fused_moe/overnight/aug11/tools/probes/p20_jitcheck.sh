#!/usr/bin/env bash
J=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
echo "=== hash dirs whose hsaco changed in the last 60 min (excluding latest) ==="
find $J -mindepth 2 -maxdepth 2 -name 'k0pf6gm_mps_mega.hsaco' -mmin -60 -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
echo
echo "=== newest 4 hashed (non-latest) ==="
ls -t $J/*/k0pf6gm_mps_mega.hsaco | grep -v '/latest/' | head -4 | xargs -r stat -c '%y %n'
echo
echo "=== jit path lines in run1 log ==="
L=$(ls -t $HOME/overnight-scratch/a11base_1_*.log | head -1)
echo "log: $L"
grep -oE 'jit/gfx950_mlx5/[^/ ]+/[A-Za-z0-9_]+\.hsaco' "$L" | sort | uniq -c | head -20
echo
echo "=== any 'mps_mega' hsaco mention ==="
grep -oE '[^ ]*k0pf6gm_mps_mega[^ ]*' "$L" | sort -u | head -10
echo
echo "=== do latest/ and the newest hash dir hold identical bytes? ==="
NEW=$(ls -t $J/*/k0pf6gm_mps_mega.hsaco | grep -v '/latest/' | head -1)
md5sum $J/latest/k0pf6gm_mps_mega.hsaco "$NEW"
echo
echo "=== build timing inside run1 (cmake lines) ==="
grep -nE 'Building|hipcc|Linking|cache hit|jit' "$L" | head -20
exit 0
