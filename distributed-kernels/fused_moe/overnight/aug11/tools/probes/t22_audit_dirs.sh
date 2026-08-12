#!/usr/bin/env bash
# t22: fingerprint the .text of EVERY mori jit build dir touched recently, so a
# mid-batch cache-key flip can be shown to be code-identical (or not).
# Table: mask 0/donor ab0c353b..., mask 4 7c4a5999..., mask 1 9d9b6539...,
#        mask 5 0fea645b...
set -uo pipefail
JIT="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
W=$HOME/overnight-scratch/e26act/audit
mkdir -p "$W"; rm -f "$W"/*.hsaco "$W"/*.elf "$W"/*.text.bin
i=0
for f in $(find "$JIT" -maxdepth 2 -name 'k0pf6gm_mps_mega.hsaco' -newermt '-60 minutes' | sort); do
  d="$(basename "$(dirname "$f")")"
  [ "$d" = "latest" ] && continue
  i=$((i+1)); cp "$f" "$W/$d.hsaco"
  echo "$d  mtime $(stat -L -c %y "$f" | cut -c1-19)"
done
echo "-- $i dir(s)"
docker exec subha_k1 bash -lc '
W=/home/subvadla/overnight-scratch/e26act/audit
OUT=/home/subvadla/overnight-scratch/e26act/out
for h in $W/*.hsaco; do
  b=$(basename $h .hsaco)
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --input=$h --output=$W/$b.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$W/$b.text.bin $W/$b.elf /dev/null 2>/dev/null
  m="?"
  for k in M0 M4 M1 M5; do
    [ -f $OUT/$k.text.bin ] && cmp -s $OUT/$k.text.bin $W/$b.text.bin && m=$k
  done
  printf "%-14s %s  %s B  -> %s\n" "$b" "$(sha256sum $W/$b.text.bin | cut -c1-16)" \
    "$(stat -c %s $W/$b.text.bin)" "$m"
done
'
exit 0
