#!/usr/bin/env bash
# exp_34: three CPU builds.
#   K  = the ARM (candidate + exp_34 C drain-retention selector)  -> resource gate
#   NC = the PROTOCOL NEGATIVE CONTROL (publish to world-1 peers) -> must compile,
#        must fail on GPU with pperr bit 25. Kept in a SEPARATE tree.
#   MK = ISA-audit build with `; E34_M14_ACQ_*` markers around the Ready=true
#        acquire, so condition 3 can be answered by POSITIVE attribution.
#   NB = MK with the coarse M8 branch disabled -> differential buffer_inv count.
set -uo pipefail
H=$HOME
D=$H/e34/DHK/distributed-kernels/fused_moe
NCT=$H/e34/negctl        # negative-control tree
MKT=$H/e34/mark          # marker tree
mkdir -p $H/e34/out

for T in "$NCT" "$MKT"; do
  rm -rf "$T"; mkdir -p "$T"
  cp -a "$H/e34/DHK/." "$T/"          # full tree so includes resolve
done

python3 - <<'PY'
import io, sys
def edit(path, subs, tag):
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if s.count(old) != 1:
            print("ANCHOR FAIL(%s) count=%d: %r" % (tag, s.count(old), old[:70])); sys.exit(1)
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", tag, path)

H = "/home/subvadla/e34/"
K = "distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"

# ---- NC: never tell the LAST peer. Everything else identical. ----------------
edit(H + "negctl/" + K, [
 ("""        for (int R = 0; R < world; ++R) {
          if (R == cur) {""",
  """        // exp_34 NEGATIVE CONTROL BUILD -- NOT AN ARM. `world - 1` leaves one
        // peer permanently untold, so its M8 must time out on the rendezvous
        // cell and set K0P6_MPS_ERR_M7DONE. If this build PASSES the gate, the
        // rendezvous is not load-bearing and the whole rung is meaningless.
        for (int R = 0; R < world - 1; ++R) {
          if (R == cur) {"""),
 ("#define K0P6_MPS_SRC_REV 28", "#define K0P6_MPS_SRC_REV 1028"),
], "NC")

# ---- MK: markers bracketing the acquire in the Ready=true instantiation -----
mk = [
 ("""  hk_moe::acquire_payload_system();
  __syncwarp();
  bool dual_mismatch = false;""",
  """  if constexpr (Ready) { asm volatile("; E34_M14_ACQ_BEGIN"); }
  hk_moe::acquire_payload_system();
  if constexpr (Ready) { asm volatile("; E34_M14_ACQ_END"); }
  __syncwarp();
  bool dual_mismatch = false;"""),
]
edit(H + "mark/" + K, mk, "MK")
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; exit 1; }

# NB tree: MK with the coarse M8 branch disabled (differential count).
rm -rf $H/e34/nobr; cp -a $H/e34/mark/. $H/e34/nobr/
python3 - <<'PY'
import io, sys
p = "/home/subvadla/e34/nobr/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
s = io.open(p, encoding="utf-8").read()
old = "    } else if (m8_coarse) {"
if s.count(old) != 1:
    print("NB anchor fail", s.count(old))
    import re
    for m in re.finditer(r".*m8_coarse.*", s):
        print(repr(m.group(0)))
    sys.exit(1)
io.open(p, "w", encoding="utf-8", newline="\n").write(
    s.replace(old, "    } else if (false && m8_coarse) {"))
print("patched NB")
PY

echo "===BUILD (subha_k1, CPU-only genco)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/e34/out
OBJ=/opt/rocm/lib/llvm/bin/llvm-objdump
BND=/opt/rocm/llvm/bin/clang-offload-bundler
[ -x $BND ] || BND=/opt/rocm/lib/llvm/bin/clang-offload-bundler
build() {  # $1=tag $2=tree-root
  T=$2
  COMMON=(--offload-arch=gfx950 -std=c++20 -O3
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
    -Rpass-analysis=kernel-resource-usage
    -I$T/include -I$T/distributed-kernels/fused_moe
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
    -I$MR -I$MR/include -I$MR/src
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
  t0=$SECONDS
  hipcc "${COMMON[@]}" --genco $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
      -o $OUT/$1.hsaco > $OUT/$1.log 2>&1
  rc=$?
  echo "### $1 exit=$rc secs=$((SECONDS-t0)) errors=$(grep -cE \"error:\" $OUT/$1.log)"
  grep -E "error:" $OUT/$1.log | head -5
  grep -A11 "Function Name: k0pf6gm_mps_mega" $OUT/$1.log | \
    grep -E "TotalSGPRs|VGPRs:|AGPRs|ScratchSize|Occupancy|Spill|LDS Size"
}
build K  $H/e34/DHK
build NC $H/e34/negctl
build MK $H/e34/mark
build NB $H/e34/nobr

echo "===ISA CENSUS (unbundle + objdump)==="
for v in K MK NB; do
  $BND --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
     -input=$OUT/$v.hsaco -output=$OUT/$v.co 2>/dev/null
  $OBJ -d --mcpu=gfx950 $OUT/$v.co > $OUT/$v.dis 2>/dev/null
  echo "$v: dis=$(wc -l < $OUT/$v.dis) mfma=$(grep -c v_mfma $OUT/$v.dis) pk_add=$(grep -c flat_atomic_pk_add_bf16 $OUT/$v.dis) inv_sc0sc1=$(grep -c \"buffer_inv sc0 sc1\" $OUT/$v.dis) scr_load=$(grep -c scratch_load $OUT/$v.dis) scr_store=$(grep -c scratch_store $OUT/$v.dis)"
done
' 2>&1
echo "===DONE==="
