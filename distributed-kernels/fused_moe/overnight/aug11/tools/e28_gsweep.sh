#!/usr/bin/env bash
# exp_28 probe 4: CPU-ONLY compile sweep of the ISOLATED M6 body (phase 1) over
# N2GM_G, to measure LDS bytes and the register tuple per G. No GPU is touched.
# Reproduces exp_59's register_fingerprint.md table and adds the LDS column,
# which that table omitted.
set -uo pipefail
SC="$HOME/overnight-scratch/e28"
mkdir -p "$SC"
cat > "$SC/gsweep_in.sh" <<'INNER'
set -uo pipefail
H=/home/subvadla
DHK=$H/Distributed-HipKittens
AM=$H/amd-master/auto-gpu-kernel/k0_fused_moe
SOL=$AM/solution/hip
MJ=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/overnight-scratch/e28
mkdir -p "$OUT"
for G in 1 2 3 4 5; do
  echo "=== N2GM_G=$G ==="
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 \
    -DN2_KERNELS_ONLY -DN2GM_G=$G \
    -Rpass-analysis=kernel-resource-usage \
    -I"$DHK/include" -I"$DHK/distributed-kernels/fused_moe" \
    -I"$SOL/hkp" -I"$AM/prefill_opt/kernels" -I"$SOL" \
    -I"$MJ" -I"$MJ/include" -I"$MJ/src" \
    "$SOL/n2_phase1_gm.cpp" -o "$OUT/p1_g$G.hsaco" 2>&1 \
    | grep -E 'remark|error' | sed 's/^.*remark: //' | head -20
done
echo "=== LDS CEILING PROBE (gfx950) ==="
for N in 163840 163841 165000; do
  printf '#include <hip/hip_runtime.h>\n__global__ void k(float* o){ __shared__ char b[%d]; b[threadIdx.x]=1; o[0]=b[0]; }\n' "$N" > "$OUT/lds$N.hip"
  echo "--- shared bytes = $N"
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 "$OUT/lds$N.hip" \
    -o "$OUT/lds$N.hsaco" 2>&1 | grep -Ei 'error|exceed|limit' | head -4
  echo "exit=$?"
done
INNER
docker exec -i subha_k1 bash -s < "$SC/gsweep_in.sh"
echo "===PROBE_DONE==="
