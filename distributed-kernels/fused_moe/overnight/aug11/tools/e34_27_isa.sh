#!/usr/bin/env bash
# exp_34 condition 3: prove a system-scope invalidate (buffer_inv sc0 sc1)
# precedes the first global_load of `slots` in the mode-14 (Ready=true)
# m8_batch instantiation. Two independent views:
#   (a) llvm-objdump -d --mcpu=gfx950 on the UNBUNDLED code object (the
#       reviewer's requested method; the .hsaco itself is an offload bundle,
#       which is why the earlier objdump attempt printed an error line);
#   (b) -S -gline-tables-only, so each buffer_inv can be attributed to a SOURCE
#       LINE and the one inside the coarse M8 branch identified positively.
set -uo pipefail
DHK=$HOME/e34/DHK
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$HOME/e34/isa; mkdir -p "$OUT"
cd "$DHK"

FLAGS="--offload-arch=gfx950 -std=c++20 -O3 -DKITTENS_CDNA4
 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -mllvm -amdgpu-mfma-vgpr-form=1
 -DK0P6GM_G=3 -DN2GM_G=3
 -I$DHK/include -I$DHK/distributed-kernels/fused_moe
 -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
 -I$MR -I$MR/include -I$MR/src
 -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include"
SRC=distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

echo "### (0) genco build"
hipcc $FLAGS --genco "$SRC" -o "$OUT/B.hsaco" 2>&1 | tail -3
ls -l "$OUT/B.hsaco"

echo "### (1) what kind of file is it"
file "$OUT/B.hsaco"
llvm-objdump --offloading "$OUT/B.hsaco" 2>&1 | head -12

echo "### (2) unbundle then objdump"
ok=0
for T in hipv4-amdgcn-amd-amdhsa--gfx950 hip-amdgcn-amd-amdhsa--gfx950 \
         hipv4-amdgcn-amd-amdhsa-unknown-gfx950; do
  if clang-offload-bundler --unbundle --type=o --targets=$T \
       -input="$OUT/B.hsaco" -output="$OUT/B.co" 2>/dev/null; then
    echo "unbundled with target: $T"; ok=1; break
  fi
done
if [ $ok -eq 0 ]; then
  echo "clang-offload-bundler failed on every target; trying objdump directly"
  cp "$OUT/B.hsaco" "$OUT/B.co"
fi
llvm-objdump -d --mcpu=gfx950 "$OUT/B.co" > "$OUT/B.dis" 2>"$OUT/B.dis.err"
echo "dis lines: $(wc -l < "$OUT/B.dis")   err: $(head -1 "$OUT/B.dis.err")"

echo "### (3) every buffer_inv in the objdump, with 8 following instructions"
grep -n "buffer_inv" "$OUT/B.dis" | head -20
echo "--- windows ---"
for L in $(grep -n "buffer_inv" "$OUT/B.dis" | cut -d: -f1 | head -12); do
  echo "===== buffer_inv at dis line $L ====="
  sed -n "$((L-3)),$((L+10))p" "$OUT/B.dis"
done

echo "### (4) line-attributed view: which SOURCE line each buffer_inv sits on"
hipcc $FLAGS -gline-tables-only --cuda-device-only -S "$SRC" -o "$OUT/B.s" 2>&1 | tail -2
echo "asm lines: $(wc -l < "$OUT/B.s")"
awk '
  /\.loc[ \t]+[0-9]+[ \t]+[0-9]+/ { split($0,a," "); f=a[2]; l=a[3] }
  /buffer_inv/                    { printf "buffer_inv  @ file %s line %s   : %s\n", f, l, $0 }
' "$OUT/B.s" | head -30
echo "--- file-number table (which file id is the .hip) ---"
grep -n "\.file[ \t]*[0-9]*" "$OUT/B.s" | head -15

echo "### (5) the coarse-M8 window in -S form: buffer_inv -> next global_load"
awk '
  /\.loc[ \t]+[0-9]+[ \t]+[0-9]+/ { split($0,a," "); l=a[3] }
  /buffer_inv/ { inv=1; n=0; printf "\n=== buffer_inv (src line %s) ===\n%s\n", l, $0; next }
  inv==1 {
     if ($0 ~ /^[ \t]*[a-z]/) { print "   " $0; n++ }
     if ($0 ~ /global_load|flat_load|buffer_load/) { print "   ^^^ FIRST LOAD after inv (src line " l ")"; inv=0 }
     if (n>40) { print "   ... (no load within 40 instrs)"; inv=0 }
  }
' "$OUT/B.s" | head -160
echo "===DONE==="
