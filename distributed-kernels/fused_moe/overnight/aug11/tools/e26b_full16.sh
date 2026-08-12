#!/usr/bin/env bash
# exp_26 follow-up step 4: the complete 16-mask truth table. Builds are ~6 s so
# there is no reason to sample the space instead of enumerating it.
set -uo pipefail
SC=$HOME/overnight-scratch/e26

docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e26
DHK=$SC/dhk
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    -DN2GM_P1_SCHED_GSCALE=${v#M} \
    $SC/tu/b1/k0pf6gm_device_tile_mps.hip -o $SC/out2/$v.hsaco > $SC/out2/$v.log 2>&1
  rc=$?
  /opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out2/$v.hsaco --output=$SC/out2/$v.elf 2>/dev/null
  /opt/rocm/llvm/bin/llvm-objcopy --dump-section=.text=$SC/out2/$v.text.bin \
    $SC/out2/$v.elf /dev/null 2>/dev/null
  /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 $SC/out2/$v.elf > $SC/out2/$v.isa 2>&1
  echo "$v exit=$rc errors=$(grep -cE "error:" $SC/out2/$v.log)"
}
export -f build_one; export SC DHK K0 MR
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do echo M$i; done \
  | xargs -P 6 -I{} bash -c "build_one {}"
' 2>&1 | sort

echo "===TEXT HASH CLASSES (D = donor build)==="
cd "$SC/out2" && sha256sum D.text.bin M0.text.bin M1.text.bin M2.text.bin M3.text.bin \
  M4.text.bin M5.text.bin M6.text.bin M7.text.bin M8.text.bin M9.text.bin M10.text.bin \
  M11.text.bin M12.text.bin M13.text.bin M14.text.bin M15.text.bin | sort

echo "===RESOURCE (scratch bytes/lane, vgpr spill)==="
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  s=$(grep 'ScratchSize' "$SC/out2/M$i.log" | sed 's/.*: //;s/ .*//')
  vs=$(grep 'VGPRs Spill' "$SC/out2/M$i.log" | sed 's/.*: //;s/ .*//')
  ss=$(grep 'SGPRs Spill' "$SC/out2/M$i.log" | sed 's/.*: //;s/ .*//')
  sg=$(grep 'TotalSGPRs' "$SC/out2/M$i.log" | sed 's/.*: //;s/ .*//')
  echo "M$i scratch=$s vgpr_spill=$vs sgpr_spill=$ss sgpr=$sg"
done

echo "===LOOP + SCRATCH TABLE==="
docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out2 && python3 ../py/mask.py D M0 M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 2>&1 | tail -25"
