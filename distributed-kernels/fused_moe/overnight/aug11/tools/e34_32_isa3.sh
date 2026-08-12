#!/usr/bin/env bash
# exp_34 condition 3, positive attribution:
#  (a) differential objdump count of `buffer_inv sc0 sc1`: K (coarse branch in)
#      vs NB (coarse branch compiled out) -- the difference IS the mode-14
#      instantiation's system-scope invalidate.
#  (b) the marker build's -S window: `; E34_M14_ACQ_BEGIN` .. first payload load,
#      which shows the invalidate sits between the markers and BEFORE the load.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/e34/out
PAT="buffer_inv sc0 sc1"

echo "=== (a) differential invalidate count ==="
for v in K MK NB; do
  n=$(grep -c "$PAT" $OUT/$v.dis)
  a=$(grep -c "buffer_inv" $OUT/$v.dis)
  echo "$v: buffer_inv total=$a   sc0 sc1 (system-scope)=$n"
done

echo "=== (b) marker build -S ==="
T=$H/e34/mark
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -I$T/include -I$T/distributed-kernels/fused_moe
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
hipcc "${COMMON[@]}" --cuda-device-only -S \
  $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip -o $OUT/MK.s 2>$OUT/MKs.log
echo "asm lines=$(wc -l < $OUT/MK.s)  markers=$(grep -c E34_M14_ACQ $OUT/MK.s)"
grep -n "E34_M14_ACQ" $OUT/MK.s

echo "=== window: BEGIN marker -> first vector memory load after it ==="
awk "
  /E34_M14_ACQ_BEGIN/ { on=1; n=0; print \"----- window start -----\" }
  on {
    if (\$0 !~ /^[[:space:]]*\\.|^[[:space:]]*;[[:space:]]*%bb|^\$/) print
    n++
    if (\$0 ~ /global_load|flat_load|buffer_load|global_atomic|flat_atomic/) {
       print \"^^^^^ FIRST vector memory op after the marker\"; on=0 }
    if (n > 60) { print \"(60 lines, no load yet)\"; on=0 }
  }
" $OUT/MK.s

echo "=== same window in the ARM binary (objdump, no markers): the 8th inv ==="
# The coarse instantiation is the sc0/sc1 invalidate present in K but absent in NB.
# Print each K window and mark which offsets exist only in K.
grep -n "$PAT" $OUT/K.dis  | awk -F: "{print \$1}" > $OUT/K.inv.lines
grep    "$PAT" $OUT/K.dis  | sed "s/.*\\/\\/ *//" | awk "{print \$1}" > $OUT/K.inv.off
grep    "$PAT" $OUT/NB.dis | sed "s/.*\\/\\/ *//" | awk "{print \$1}" > $OUT/NB.inv.off
echo "K offsets:";  cat $OUT/K.inv.off  | tr "\n" " "; echo
echo "NB offsets:"; cat $OUT/NB.inv.off | tr "\n" " "; echo
' 2>&1
echo "===DONE==="
