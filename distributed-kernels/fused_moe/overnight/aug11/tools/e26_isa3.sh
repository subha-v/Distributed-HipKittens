#!/usr/bin/env bash
# exp_26 step 3c: unbundle -> device ELF hash (the real B1==B0 gate) -> disasm.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
SC=/home/subvadla/overnight-scratch/e26
for v in B0 B1 B2; do
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.gfx950.elf
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $SC/out/$v.gfx950.elf \
    | tail -n +3 > $SC/out/$v.isa
done
echo "=== DEVICE ELF HASHES (this is the B1==B0 gate) ==="
sha256sum $SC/out/*.gfx950.elf
ls -la $SC/out/*.gfx950.elf
echo "=== ISA COUNTS ==="
for v in B0 B1 B2; do
  echo "$v lines=$(wc -l < $SC/out/$v.isa) mfma=$(grep -c v_mfma $SC/out/$v.isa) scratch=$(grep -c scratch_ $SC/out/$v.isa) waitcnt=$(grep -c s_waitcnt $SC/out/$v.isa) barrier=$(grep -c s_barrier $SC/out/$v.isa)"
done
echo "=== ISA TEXT DIFF B0 vs B1 ==="
diff $SC/out/B0.isa $SC/out/B1.isa > $SC/out/b0b1.isadiff 2>&1
echo "b0_b1_isa_difflines=$(wc -l < $SC/out/b0b1.isadiff)"
head -20 $SC/out/b0b1.isadiff
echo "=== ISA TEXT DIFF B1 vs B2 ==="
diff $SC/out/B1.isa $SC/out/B2.isa > $SC/out/b1b2.isadiff 2>&1
echo "b1_b2_isa_difflines=$(wc -l < $SC/out/b1b2.isadiff)"
echo "=== MFMA VARIANTS ==="
grep -oE "v_mfma_[a-z0-9_]+" $SC/out/B1.isa | sort | uniq -c
echo "=== FORMAT SAMPLE (first mfma +- context) ==="
f=$(grep -n v_mfma $SC/out/B1.isa | head -1 | cut -d: -f1)
awk -v a=$((f-8)) -v b=$((f+4)) "NR>=a && NR<=b" $SC/out/B1.isa
echo "=== BRANCH SAMPLE ==="
grep -nE "s_cbranch|s_branch" $SC/out/B1.isa | head -8
echo "=== LABEL SAMPLE ==="
grep -nE "^[0-9a-f]{16} <" $SC/out/B1.isa | head -8
'
