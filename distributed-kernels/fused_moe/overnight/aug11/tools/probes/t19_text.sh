#!/usr/bin/env bash
# t19: the two mid-batch JIT builds differ in hsaco bytes at the same size.
# Compare their .text sections -- if those match, the CODE is identical and
# `hsaco_before != hsaco_after` is NOT evidence of a code change.
set -uo pipefail
JIT="$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5"
W=$HOME/overnight-scratch/e26act/jitcmp
mkdir -p "$W"
cp "$JIT/9df63b03fb16/k0pf6gm_mps_mega.hsaco" "$W/A.hsaco"
cp "$JIT/b43b280256c6/k0pf6gm_mps_mega.hsaco" "$W/B.hsaco"
echo "== first difference, in context =="
cmp -l "$W/A.hsaco" "$W/B.hsaco" | head -20
echo "   (differing byte count: $(cmp -l "$W/A.hsaco" "$W/B.hsaco" | wc -l) of $(stat -c %s "$W/A.hsaco"))"
docker exec subha_k1 bash -lc '
set -uo pipefail
W=/home/subvadla/overnight-scratch/e26act/jitcmp
for v in A B; do
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$W/$v.hsaco --output=$W/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$W/$v.text.bin $W/$v.elf /dev/null 2>/dev/null
done
echo "== .text sha256 / size =="
for v in A B; do
  [ -f $W/$v.text.bin ] && printf "%s %s  %s B\n" "$v" \
    "$(sha256sum $W/$v.text.bin | cut -d" " -f1)" "$(stat -c %s $W/$v.text.bin)"
done
if cmp -s $W/A.text.bin $W/B.text.bin; then
  echo "TEXT IDENTICAL -- same code, cache key moved only"
else
  echo "TEXT DIFFERS -- the arm really changed mid-batch"
fi
echo "== what the differing bytes are (strings around the delta) =="
llvm-readelf --string-dump=.comment $W/A.elf 2>/dev/null | head -5
llvm-readelf --string-dump=.comment $W/B.elf 2>/dev/null | head -5
echo "== CPU-side M0 .text for reference =="
sha256sum /home/subvadla/overnight-scratch/e26act/out/M0.text.bin 2>/dev/null
'
exit 0
