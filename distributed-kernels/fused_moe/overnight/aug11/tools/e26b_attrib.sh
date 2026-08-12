#!/usr/bin/env bash
# exp_26 follow-up step 6: attribute the 99 migrated scratch accesses to source
# lines. -gline-tables-only must not perturb -O3 codegen; that is checked by
# re-hashing .text against the non-debug build of the same mask.
set -uo pipefail

docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e26
DHK=$SC/dhk
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
for v in 0 1; do
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 -gline-tables-only \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    -DN2GM_P1_SCHED_GSCALE=$v \
    $SC/tu/b1/k0pf6gm_device_tile_mps.hip -o $SC/out2/G$v.hsaco > $SC/out2/G$v.log 2>&1
  echo "G$v exit=$?"
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out2/G$v.hsaco --output=$SC/out2/G$v.elf 2>/dev/null
  /opt/rocm/llvm/bin/llvm-objcopy --dump-section=.text=$SC/out2/G$v.text.bin $SC/out2/G$v.elf /dev/null 2>/dev/null
  /opt/rocm/llvm/bin/llvm-objdump -d -l --mcpu=gfx950 $SC/out2/G$v.elf > $SC/out2/G$v.lisa 2>&1
done
echo "--- codegen unperturbed by -g? (G0 vs M0, G1 vs M1) ---"
sha256sum $SC/out2/G0.text.bin $SC/out2/M0.text.bin $SC/out2/G1.text.bin $SC/out2/M1.text.bin
'

docker exec subha_k1 bash -lc '
cd /home/subvadla/overnight-scratch/e26/out2
python3 - <<"PYEOF"
import re, collections
def scan(path):
    cur = None
    rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"^; (\S+):(\d+)", line)
        if m: cur = (m.group(1).split("/")[-1], int(m.group(2))); continue
        m = re.match(r"^\t(\S+)\s.*//\s*([0-9A-F]+):", line)
        if m and m.group(1).startswith("scratch_"):
            rows.append((int(m.group(2),16), m.group(1), cur))
    return rows
for tag in ("G0","G1"):
    rows = scan(f"{tag}.lisa")
    print(f"===== {tag} ({len(rows)} scratch accesses) =====")
    c = collections.Counter(r[2] for r in rows)
    for site, n in c.most_common(14):
        print(f"   {n:4d}  {site}")
PYEOF
'
