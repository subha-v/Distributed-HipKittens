#!/usr/bin/env bash
# exp_26 step 3f: explain the 41 residual ELF bytes; pull literal ISA excerpts.
set -uo pipefail
docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
echo "=== SECTION MAP (B0) ==="
/opt/rocm/llvm/bin/llvm-readelf -S $SC/out/B0.gfx950.elf | awk "/Name|PROGBITS|NOTE|STRTAB|SYMTAB|NOBITS|RELA|HASH|DYNAMIC/"
echo "=== which sections hold the 41 differing offsets? ==="
cmp -l $SC/out/B0.gfx950.elf $SC/out/B1.gfx950.elf | awk "{print \$1}" > /tmp/d.txt
python3 - <<PY
offs=[int(x)-1 for x in open("/tmp/d.txt")]
print("n_diff_bytes",len(offs))
print("min",hex(min(offs)),"max",hex(max(offs)))
import collections
print(sorted(hex(o) for o in offs)[:50])
PY
echo "=== .note payload identical? ==="
for v in B0 B1; do /opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=.note $SC/out/$v.gfx950.elf /tmp/$v.note.bin 2>/dev/null; done
sha256sum /tmp/B0.note.bin /tmp/B1.note.bin 2>/dev/null || echo "(no .note section by that name)"
echo "=== all section hashes B0 vs B1 ==="
for s in .text .rodata .note .dynsym .dynstr .hash .comment .debug_info; do
  for v in B0 B1; do
    /opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=$s $SC/out/$v.gfx950.elf /tmp/$v$s.bin 2>/dev/null
  done
  a=$(sha256sum /tmp/B0$s.bin 2>/dev/null | cut -c1-16); b=$(sha256sum /tmp/B1$s.bin 2>/dev/null | cut -c1-16)
  sz=$(stat -c%s /tmp/B0$s.bin 2>/dev/null || echo -)
  [ -n "$a" ] && echo "  $s size=$sz B0=$a B1=$b $( [ "$a" = "$b" ] && echo SAME || echo DIFF )"
done

echo
echo "=== B1 phase-1 K-loop: the vmcnt(0) site, verbatim (+-6) ==="
grep -n "" $SC/out/B1.isa | sed -n "$(grep -n "0000B2AC" $SC/out/B1.isa | cut -d: -f1 | head -1 | awk "{print \$1-6\",\"\$1+6}")p"
echo
echo "=== B2 phase-1 K-loop: the vmcnt(0) site, verbatim (+-6) ==="
grep -n "" $SC/out/B2.isa | sed -n "$(grep -n "0000BD00" $SC/out/B2.isa | cut -d: -f1 | head -1 | awk "{print \$1-6\",\"\$1+6}")p"
echo
echo "=== B2 first barrier site, verbatim (+-5) ==="
grep -n "" $SC/out/B2.isa | sed -n "$(grep -n "0000B9F4" $SC/out/B2.isa | cut -d: -f1 | head -1 | awk "{print \$1-5\",\"\$1+5}")p"
echo
echo "=== B1 first barrier site, verbatim (+-5) ==="
grep -n "" $SC/out/B1.isa | sed -n "$(grep -n "0000B79C" $SC/out/B1.isa | cut -d: -f1 | head -1 | awk "{print \$1-5\",\"\$1+5}")p"
'
