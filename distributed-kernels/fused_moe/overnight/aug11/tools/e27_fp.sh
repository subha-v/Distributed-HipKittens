#!/usr/bin/env bash
# exp_27: fingerprint the mps_mega build that was ACTUALLY LAUNCHED, by .text.
#
# `hsaco_before != hsaco_after` is not evidence of a code change -- a "rebuild"
# tonight differed in 36 of 188,744 bytes (embedded build paths) with identical
# .text. Only the == direction is sound. So the arm is identified by the sha256
# of the .text section of the JIT hsaco the run resolved, which is invariant to
# build paths and to the JIT's content-hash directory name.
set -uo pipefail
JITDIR=$HOME/.cache/k0-mok-synthetic-prefill/mori/jit/gfx950_mlx5
W=$HOME/overnight-scratch/e27/fp
mkdir -p "$W"

link="$JITDIR/latest/k0pf6gm_mps_mega.hsaco"
if [ -e "$link" ]; then f="$(readlink -f "$link")"; else
  f="$(ls -tL "$JITDIR"/*/k0pf6gm_mps_mega.hsaco 2>/dev/null | head -1)"; fi
[ -n "$f" ] && [ -e "$f" ] || { echo "no mps hsaco found under $JITDIR"; exit 1; }

echo "resolved : $f"
echo "jit dir  : $(basename "$(dirname "$f")")"
echo "mtime    : $(stat -L -c %y "$f" | cut -c1-19)"
echo "size     : $(stat -L -c %s "$f") B"
echo "file sha : $(sha256sum "$f" | cut -d' ' -f1)"

rm -f "$W/j.elf" "$W/j.text.bin"
docker exec subha_k1 bash -lc "
  set -uo pipefail
  W=/home/subvadla/overnight-scratch/e27/fp
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input='$f' --output=\$W/j.elf 2>/dev/null || cp '$f' \$W/j.elf
  llvm-objcopy --dump-section=.text=\$W/j.text.bin \$W/j.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 \$W/j.elf > \$W/j.isa 2>/dev/null
" 2>&1 | tail -3

if [ -f "$W/j.text.bin" ]; then
  echo "TEXT sha : $(sha256sum "$W/j.text.bin" | cut -d' ' -f1)"
  echo "TEXT size: $(stat -c %s "$W/j.text.bin") B"
else
  echo "TEXT sha : (could not extract)"
fi

# The arm-discriminating immediate: the gather's loop limit. 0x13ff = 5376-257 is
# the group-major arm (trip 21); 0x43f = 1344-257 is the token-major arm (trip 6).
if [ -f "$W/j.isa" ]; then
  n21=$(grep -cE 's_movk_i32 s[0-9]+, 0x13ff' "$W/j.isa")
  n6=$(grep -cE 's_movk_i32 s[0-9]+, 0x43f' "$W/j.isa")
  x4=$(grep -cE '^[[:space:]]+flat_load_dwordx4' "$W/j.isa")
  w2=$(grep -cE '^[[:space:]]+ds_write2_b32' "$W/j.isa")
  mf=$(grep -cE '^[[:space:]]+v_mfma' "$W/j.isa")
  echo "ARM PROBE: 0x13ff(group-major,trip21)=$n21  0x43f(token-major,trip6)=$n6  dwordx4=$x4  ds_write2_b32=$w2  mfma=$mf"
  if [ "$n6" -gt 0 ] && [ "$n21" -eq 0 ]; then echo "  => launched arm = 1 (TOKEN-MAJOR)"
  elif [ "$n21" -gt 0 ] && [ "$n6" -eq 0 ]; then echo "  => launched arm = 0 (GROUP-MAJOR / control)"
  else echo "  => AMBIGUOUS -- do not trust this batch's arm attribution"; fi
fi
exit 0
