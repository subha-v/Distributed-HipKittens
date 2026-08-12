#!/usr/bin/env bash
# exp_34 condition 4, CORRECTED: the first negative-control patch hit the PARITY
# per-row publish loop (modes 0/1/5/6), not mode 14's rendezvous publish -- both
# loops read `for (int R = 0; R < world; ++R) { if (R == cur) {`. Re-cut the
# control against the `self_slot` publish, which is mode-14-only.
set -uo pipefail
H=$HOME
D=$H/e34/DHK/distributed-kernels/fused_moe
NCT=$H/e34/negctl
rm -rf "$NCT"; mkdir -p "$NCT"; cp -a "$H/e34/DHK/." "$NCT/"

echo "############ the mode-14 publish site, verbatim ############"
sed -n '1755,1782p' "$D/k0pf6gm_device_tile_mps.hip"

python3 - <<'PY'
import io, sys, re
p = "/home/subvadla/e34/negctl/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
s = io.open(p, encoding="utf-8").read()

# Locate the mode-14 publish loop by its UNIQUE body (self_slot), then rewrite
# only that loop's bound.
m = re.search(r"( *)for \(int R = 0; R < world; \+\+R\) \{\n(?:.*\n){0,8}?"
              r".*publish_epoch<hk_moe::scope::agent>\(self_slot, epoch32\);", s)
if not m:
    print("NC ANCHOR FAIL: no self_slot publish loop found"); sys.exit(1)
head = m.group(0)
if head.count("for (int R = 0; R < world; ++R)") != 1:
    print("NC ANCHOR FAIL: ambiguous"); sys.exit(1)
if s.count(head) != 1:
    print("NC ANCHOR FAIL: not unique in file"); sys.exit(1)
ind = m.group(1)
new = head.replace(
    "for (int R = 0; R < world; ++R) {",
    "// exp_34 PROTOCOL NEGATIVE CONTROL BUILD -- NOT AN ARM, NEVER A NUMBER.\n"
    + ind + "// `world - 1` leaves the LAST rank untold: nobody publishes into\n"
    + ind + "// rank world-1's segment, so all 8 of its M7-done polls must time\n"
    + ind + "// out and set K0P6_MPS_ERR_M7DONE (bit 25) on that rank alone, its\n"
    + ind + "// M8 must be skipped by the payload_ok mask, and [MOK GATE] must\n"
    + ind + "// fail. If this build PASSES, the rendezvous is not load-bearing\n"
    + ind + "// and the entire mode-14 rung is meaningless.\n"
    + ind + "for (int R = 0; R < world - 1; ++R) {")
s = s.replace(head, new)
s = s.replace("#define K0P6_MPS_SRC_REV 28", "#define K0P6_MPS_SRC_REV 1028")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("NC patched (self_slot loop)")
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; exit 1; }

echo "############ the control diff (must touch ONLY the self_slot loop + rev) ############"
diff -u "$D/k0pf6gm_device_tile_mps.hip" \
        "$NCT/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" \
  | tee "$H/e34/out/negctl.patch"

echo "############ rebuild the control ############"
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
OUT=$H/e34/out; T=$H/e34/negctl
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$T/include -I$T/distributed-kernels/fused_moe
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
hipcc "${COMMON[@]}" --genco $T/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip \
   -o $OUT/NC.hsaco > $OUT/NC.log 2>&1
echo "NC exit=$? errors=$(grep -cE \"error:\" $OUT/NC.log)"
grep -A11 "Function Name: k0pf6gm_mps_mega" $OUT/NC.log | \
  grep -E "TotalSGPRs|VGPRs:|AGPRs|ScratchSize|Occupancy|Spill|LDS Size" | \
  sed "s/.*remark: *//;s/ \[-Rpass.*//"
ls -l $OUT/NC.hsaco; sha256sum $OUT/NC.hsaco
' 2>&1
echo "===DONE==="
