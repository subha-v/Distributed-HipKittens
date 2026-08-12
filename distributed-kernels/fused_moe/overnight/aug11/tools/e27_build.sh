#!/usr/bin/env bash
# exp_27 BUILD GATES. CPU-only (hipcc --genco in subha_k1). No GPU work.
#
# Three TUs, each in its OWN directory with its own n2_phase1_gm_mps.cpp, so the
# quoted #include resolves TU-locally and the node checkout is never touched:
#   R  = the ratchet source at f34e72fc (no knob at all)          -- the reference
#   A0 = exp_27 source, K0P6_MPS_ASCALE_TM 0                      -- control arm
#   A1 = exp_27 source, K0P6_MPS_ASCALE_TM 1                      -- candidate arm
#
# A1 is produced by rewriting the HASHED LITERAL in the .hip, not by -D, so the
# TU that gets gated is byte-for-byte the TU that gets committed and launched.
#
# GATES
#   G1  .text(A0) == .text(R)         -- the knob is inert at 0; the control arm
#                                        IS the ratchet binary
#   G2  A1 resource tuple             -- AGPR/ArchVGPR/LDS exact, scratch <= 128
#   G3  MFMA census 180 (96+84)       -- the K-loops are untouched
#   G5  gather ISA                    -- dwordx4 present, trip count 6 not 21
#   G7  sc_dst dead-store check       -- the M5 transpose really is gone
set -uo pipefail
SC=$HOME/overnight-scratch/e27
DHK=$HOME/Distributed-HipKittens
FM=$DHK/distributed-kernels/fused_moe
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
IN=$SC/incoming
mkdir -p "$SC/tu/R" "$SC/tu/A0" "$SC/tu/A1" "$SC/out"
rm -f "$SC/out/"*

echo "===INPUTS==="
for f in k0pf6gm_device_tile_mps.hip n2_phase1_gm_mps.cpp \
         ratchet_k0pf6gm_device_tile_mps.hip ratchet_n2_phase1_gm_mps.cpp; do
  [ -f "$IN/$f" ] || { echo "MISSING $IN/$f"; exit 2; }
  printf '%-42s %s  %s lines\n' "$f" "$(sha256sum "$IN/$f" | cut -c1-16)" "$(wc -l < "$IN/$f")"
done
echo "node checkout .hip (must be the ratchet): $(sha256sum "$FM/k0pf6gm_device_tile_mps.hip" | cut -c1-16)"
echo "node checkout p1   (must be the ratchet): $(sha256sum "$FM/n2_phase1_gm_mps.cpp" | cut -c1-16)"

echo
echo "===LAY OUT THE THREE TUs==="
cp "$IN/ratchet_k0pf6gm_device_tile_mps.hip" "$SC/tu/R/k0pf6gm_device_tile_mps.hip"
cp "$IN/ratchet_n2_phase1_gm_mps.cpp"        "$SC/tu/R/n2_phase1_gm_mps.cpp"
# sanity: the extracted ratchet must equal the node checkout, or my `git show`
# and the node's tree disagree and the reference is worthless.
cmp -s "$SC/tu/R/k0pf6gm_device_tile_mps.hip" "$FM/k0pf6gm_device_tile_mps.hip" \
  && echo "R .hip == node checkout .hip  OK" || echo "R .hip != node checkout .hip  ** FATAL **"
cmp -s "$SC/tu/R/n2_phase1_gm_mps.cpp" "$FM/n2_phase1_gm_mps.cpp" \
  && echo "R p1   == node checkout p1    OK" || echo "R p1   != node checkout p1    ** FATAL **"

cp "$IN/k0pf6gm_device_tile_mps.hip" "$SC/tu/A0/k0pf6gm_device_tile_mps.hip"
cp "$IN/n2_phase1_gm_mps.cpp"        "$SC/tu/A0/n2_phase1_gm_mps.cpp"
sed 's/^#define K0P6_MPS_ASCALE_TM 0$/#define K0P6_MPS_ASCALE_TM 1/' \
  "$SC/tu/A0/k0pf6gm_device_tile_mps.hip" > "$SC/tu/A1/k0pf6gm_device_tile_mps.hip"
cp "$IN/n2_phase1_gm_mps.cpp"        "$SC/tu/A1/n2_phase1_gm_mps.cpp"

echo "--- A0 vs A1 TU diff (must be exactly the one literal) ---"
diff "$SC/tu/A0/k0pf6gm_device_tile_mps.hip" "$SC/tu/A1/k0pf6gm_device_tile_mps.hip"
echo "--- knob literals + SRC_REV per TU ---"
for v in R A0 A1; do
  printf '%-3s SRC_REV=%s  ASCALE_TM=%s\n' "$v" \
    "$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')" \
    "$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip" | awk '{print $3}' || echo '(absent)')"
done

echo
echo "===BUILD (subha_k1, CPU-only genco, 3-way)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e27
DHK=$H/Distributed-HipKittens
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1
  t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    -I$SC/tu/$v \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $SC/out/$v.hsaco > $SC/out/$v.log 2>&1
  rc=$?
  echo "$v exit=$rc secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out/$v.log)"
}
export -f build_one; export SC DHK K0 MR
printf "%s\n" R A0 A1 | xargs -P 3 -I{} bash -c "build_one {}"
echo "===UNBUNDLE + .text==="
for v in R A0 A1; do
  [ -f $SC/out/$v.hsaco ] || { echo "$v MISSING HSACO"; continue; }
  clang-offload-bundler --type=o --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input=$SC/out/$v.hsaco --output=$SC/out/$v.elf 2>/dev/null
  llvm-objcopy --dump-section=.text=$SC/out/$v.text.bin $SC/out/$v.elf /dev/null 2>/dev/null
  llvm-objdump -d --mcpu=gfx950 $SC/out/$v.elf > $SC/out/$v.isa 2>/dev/null
done
' 2>&1 | tail -20

echo
echo "===COMPILE ERRORS (if any)==="
for v in R A0 A1; do
  n=$(grep -cE "error:" "$SC/out/$v.log" 2>/dev/null || echo 0)
  if [ "$n" != "0" ]; then echo "---- $v ($n) ----"; grep -E "error:" "$SC/out/$v.log" | head -12; fi
done
echo "(no output above == all three compiled clean)"

echo
echo "===G1: .text sha256 + size==="
for v in R A0 A1; do
  if [ -f "$SC/out/$v.text.bin" ]; then
    printf '%-3s %s  %s B\n' "$v" "$(sha256sum "$SC/out/$v.text.bin" | cut -d' ' -f1)" \
      "$(stat -c %s "$SC/out/$v.text.bin")"
  else printf '%-3s (no .text)\n' "$v"; fi
done
if cmp -s "$SC/out/R.text.bin" "$SC/out/A0.text.bin"; then
  echo "G1 PASS: arm 0 .text is BYTE-IDENTICAL to the ratchet -- the knob is inert at 0"
else
  echo "G1 FAIL: arm 0 .text DIFFERS from the ratchet"
  cmp "$SC/out/R.text.bin" "$SC/out/A0.text.bin" | head -3
fi
if cmp -s "$SC/out/A0.text.bin" "$SC/out/A1.text.bin"; then
  echo "** A1 .text == A0 .text -- THE CANDIDATE DID NOT COMPILE DIFFERENTLY, STOP **"
else
  echo "OK: A1 .text differs from A0 (the arm really changed the code)"
fi

echo
echo "===G2: resource tuple (k0pf6gm_mps_mega)==="
for v in R A0 A1; do
  echo "---- $v ----"
  grep -E 'Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size|Spill' \
    "$SC/out/$v.log" 2>/dev/null | sed 's/^remark: //' | \
    awk '/k0pf6gm_mps_mega/{p=1} p' | head -14
done

echo
echo "===G3/G4: census + scratch ops==="
for v in R A0 A1; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '%-3s mfma=%s  scratch_ops=%s  pk_add_bf16=%s  s_barrier=%s\n' "$v" \
    "$(grep -cE '^\s+v_mfma' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+scratch_(load|store)' "$SC/out/$v.isa")" \
    "$(grep -c 'flat_atomic_pk_add_bf16' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+s_barrier' "$SC/out/$v.isa")"
done

echo
echo "===G5: load-mix delta (whole kernel, static)==="
for v in R A0 A1; do
  [ -f "$SC/out/$v.isa" ] || continue
  printf '%-3s gl_dword=%s gl_dwordx2=%s gl_dwordx4=%s ds_write_b32=%s ds_write_b64=%s ds_write_b128=%s\n' "$v" \
    "$(grep -cE '^\s+global_load_dword ' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+global_load_dwordx2' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+global_load_dwordx4' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+ds_write_b32' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+ds_write_b64' "$SC/out/$v.isa")" \
    "$(grep -cE '^\s+ds_write_b128' "$SC/out/$v.isa")"
done

echo
echo "===G7: does 'sc_dst' still have a live store path in A1? (source-level) ==="
grep -n 'scale_transpose_row\|zero_part_scale_transpose' "$SC/tu/A1/k0pf6gm_device_tile_mps.hip"
echo "--- and which descriptor slot phase-1 is handed, per arm ---"
for v in R A0 A1; do
  echo "-- $v --"
  grep -n -A3 'n2p6gm_phase1_body(' "$SC/tu/$v/k0pf6gm_device_tile_mps.hip" | grep -E 'SC_DST|SC_STAGE|#if|#else'
done
echo "===DONE==="
exit 0
