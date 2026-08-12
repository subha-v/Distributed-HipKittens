#!/usr/bin/env bash
# exp_26 follow-up step 2: disassemble the seven builds, extract .text, and
# emit the per-mask subset table (resource tuple, scratch split, drain position,
# barrier partition).
set -uo pipefail
SC=$HOME/overnight-scratch/e26
V="D M0 M2 M4 M6 M9 M15"

docker exec subha_k1 bash -lc "
SC=/home/subvadla/overnight-scratch/e26
for v in $V; do
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 \$SC/out2/\$v.hsaco > \$SC/out2/\$v.isa 2>&1
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=\$SC/out2/\$v.hsaco --output=\$SC/out2/\$v.elf 2>/dev/null
  /opt/rocm/llvm/bin/llvm-objcopy --dump-section=.text=\$SC/out2/\$v.text.bin \
    \$SC/out2/\$v.elf /dev/null 2>/dev/null
done
echo objdump_done
"

echo "===ARTIFACT SIZES==="
ls -la "$SC/out2/"*.hsaco "$SC/out2/"*.text.bin | awk '{print $5, $9}'

echo "===TEXT HASHES (D is the donor build; M0 must match it)==="
sha256sum "$SC/out2/"*.text.bin
echo "--- first-run B0 .text for cross-run continuity ---"
sha256sum "$SC/out/B0.text.bin" "$SC/out/B2.text.bin" 2>/dev/null

echo "===RESOURCE TUPLES==="
for v in $V; do
  echo "--- $v ---"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' "$SC/out2/$v.log" \
    | sed 's|.*remark: *||' | sed 's|\[-Rpass-analysis=kernel-resource-usage\]||'
done
