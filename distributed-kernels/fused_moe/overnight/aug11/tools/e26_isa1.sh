#!/usr/bin/env bash
# exp_26 step 3a: disassemble + resource tuples + format recon.
set -uo pipefail
SC=$HOME/overnight-scratch/e26

docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
for v in B0 B1 B2; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $SC/out/$v.hsaco > $SC/out/$v.isa 2>&1
  /opt/rocm/llvm/bin/llvm-readelf --notes $SC/out/$v.hsaco > $SC/out/$v.notes 2>&1
done
echo objdump_done
'

echo "===RESOURCE TUPLES==="
for v in B0 B1 B2; do
  echo "--- $v ---"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' "$SC/out/$v.log" \
    | sed 's|.*remark: *||' | sed 's|\[-Rpass-analysis=kernel-resource-usage\]||'
done

echo "===ISA SIZE / TOTAL MFMA==="
for v in B0 B1 B2; do
  n=$(grep -c 'v_mfma' "$SC/out/$v.isa")
  l=$(wc -l < "$SC/out/$v.isa")
  s=$(grep -c 'scratch_' "$SC/out/$v.isa")
  echo "$v isa_lines=$l v_mfma=$n scratch_ops=$s"
done

echo "===B0 vs B1 RAW ISA DIFF (first 40 lines)==="
diff "$SC/out/B0.isa" "$SC/out/B1.isa" | head -40
echo "b0_b1_difflines=$(diff "$SC/out/B0.isa" "$SC/out/B1.isa" | wc -l)"

echo "===B1 vs B2 ISA DIFF SIZE==="
echo "b1_b2_difflines=$(diff "$SC/out/B1.isa" "$SC/out/B2.isa" | wc -l)"

echo "===OBJDUMP FORMAT SAMPLE (around first mfma)==="
grep -n 'v_mfma' "$SC/out/B1.isa" | head -3
awk 'NR>=1 && NR<=25' "$SC/out/B1.isa"
echo "..."
first=$(grep -n 'v_mfma' "$SC/out/B1.isa" | head -1 | cut -d: -f1)
awk -v a=$((first-12)) -v b=$((first+6)) 'NR>=a && NR<=b' "$SC/out/B1.isa"

echo "===BRANCH FORMAT SAMPLE==="
grep -nE 's_cbranch|s_branch' "$SC/out/B1.isa" | head -6

echo "===KERNEL SYMBOLS==="
grep -nE '^[0-9a-f]+ <' "$SC/out/B1.isa" | head -20
