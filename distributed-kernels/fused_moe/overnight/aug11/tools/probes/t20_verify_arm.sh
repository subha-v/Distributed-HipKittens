#!/usr/bin/env bash
# t20: prove WHICH mask actually ran. Unbundle the mps_mega hsaco the last run
# loaded and print its .text sha256, to be matched against the CPU-side table:
#   mask 0 / donor  ab0c353bef065898881b7f845f44c20085bf5c8661abaa0ca8fc19988046f43a  179904
#   mask 4          7c4a5999e725c9136c56aa0f4d517403e7e2347716267215474679f0c9ec3598  179904
#   mask 1          9d9b6539c2e179e963c8fba652ea5fc453923207d3db29d5e518cfec12f81c91  180608
#   mask 5          0fea645b97c934ec63808b70acf3d3d8653f41c0b8804c612846d2a207d36a80  180608
# Needed because the hsaco bytes carry embedded build paths: two builds of the
# SAME code hash differently, so hsaco identity cannot answer this.
set -uo pipefail
JIT="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
W=$HOME/overnight-scratch/e26act/armchk
mkdir -p "$W"; rm -f "$W"/*
L="$JIT/latest/k0pf6gm_mps_mega.hsaco"
R="$(readlink -f "$L")"
echo "latest -> $R"
stat -L -c 'mtime %y  size %s' "$R"
cp "$R" "$W/L.hsaco"
docker exec subha_k1 bash -lc '
W=/home/subvadla/overnight-scratch/e26act/armchk
clang-offload-bundler --type=o --unbundle \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  --input=$W/L.hsaco --output=$W/L.elf 2>/dev/null
llvm-objcopy --dump-section=.text=$W/L.text.bin $W/L.elf /dev/null 2>/dev/null
printf "LOADED .text %s  %s B\n" "$(sha256sum $W/L.text.bin | cut -d" " -f1)" "$(stat -c %s $W/L.text.bin)"
for m in D M0 M4 M1 M5; do
  f=/home/subvadla/overnight-scratch/e26act/out/$m.text.bin
  [ -f "$f" ] && cmp -s "$f" $W/L.text.bin && echo "  == MATCHES CPU-side build $m"
done
'
exit 0
