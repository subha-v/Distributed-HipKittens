#!/usr/bin/env bash
# exp_26 follow-up step 2b: objdump the UNBUNDLED device ELF (the bundle itself
# is not a valid object file -- same trap as the first run).
set -uo pipefail
docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
for v in D M0 M2 M4 M6 M9 M15; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $SC/out2/$v.elf > $SC/out2/$v.isa 2>&1
  echo "$v isa_lines=$(wc -l < $SC/out2/$v.isa) mfma=$(grep -c v_mfma $SC/out2/$v.isa) scratch=$(grep -c scratch_ $SC/out2/$v.isa)"
done
'
