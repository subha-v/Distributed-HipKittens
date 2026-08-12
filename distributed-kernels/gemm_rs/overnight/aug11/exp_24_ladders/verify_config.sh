#!/usr/bin/env bash
# exp_24 re-measure, PHASE 0c: prove the config change reached the binary.
# CPU only -- no GPU, no lease. Three independent checks, because "a config
# change that did not reach codegen" is the second way this re-measure comes
# back wrong (the first being a stale build, ruled out in go_prep2.sh).
#
#   A. shape geometry from the LIVE plan (dhk_rt.resolve_shape, host arithmetic
#      only): tiles / producers / tiles_per_cta, and the rgroup the shipped rule
#      selects for each graded row. Must read 1/1/1/1/2/4 with shape 5 == 2.
#      This is exp_26's ab_pershape.py check (expected_rgroup + _geometry).
#   B. codegen reachability: recompile the SAME source, SAME module name, with
#      PERSHAPE=0, and require the object to DIFFER from the shipped one. If
#      PERSHAPE=2 were dead code the two would be byte-identical.
#   C. the shipped source default, re-read at measurement time.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
GEMM=/home/subvadla/dhk/distributed-kernels/gemm_rs
SHIPPED=$ON/harness/build/gemm_rs_mi300x.so

echo "=============== A. geometry + rgroup from the live plan ==============="
docker exec -w "$ON/harness" dhk-gemmrs python3 - <<'PY'
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location(
    "dhk_rt", "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/build/dhk_rt.so")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)

SCORED = [(64, 7168, 18432, True), (512, 4096, 12288, False),
          (2048, 4096, 12288, False), (4096, 4096, 12288, False),
          (8192, 4096, 14336, False), (8192, 8192, 29568, False)]
# The shipped rule at PERSHAPE=2: a descending select over compile-time
# literals {RELEASE_GROUP, 2, 1} with RELEASE_GROUP=4 and FULL_ONLY=1.
def shipped_rgroup(ppc, RG=4):
    if ppc >= RG: return RG
    if ppc >= 2:  return 2
    return 1
def incumbent_rgroup(ppc, RG=4):
    return RG if ppc >= RG else 1

print(f"  {'shape':>22} {'row':>4} {'tiles':>6} {'prod':>5} {'t/CTA':>6} "
      f"{'was':>4} {'now':>4}  changed")
expect = [1, 1, 1, 1, 2, 4]
ok = True
for i, (m, n, k, bias) in enumerate(SCORED):
    p = rt.resolve_shape(m, n, k, bias)
    tiles, prod = int(p["gemm_tiles"]), int(p["num_gemm_ctas"])
    ppc = -(-tiles // prod)
    was, now = incumbent_rgroup(ppc), shipped_rgroup(ppc)
    flag = "<== ONLY ROW THAT MOVES" if was != now else ""
    print(f"  {m}x{n}x{k:>5} {int(p['config_row']):>4} {tiles:>6} {prod:>5} "
          f"{ppc:>6} {was:>4} {now:>4}  {flag}")
    if now != expect[i]:
        print(f"  FATAL shape {i+1}: rgroup {now} != expected {expect[i]}"); ok = False
moved = [i + 1 for i, (m, n, k, b) in enumerate(SCORED)
         if incumbent_rgroup(-(-int(rt.resolve_shape(m,n,k,b)['gemm_tiles'])
                              // int(rt.resolve_shape(m,n,k,b)['num_gemm_ctas'])))
         != shipped_rgroup(-(-int(rt.resolve_shape(m,n,k,b)['gemm_tiles'])
                             // int(rt.resolve_shape(m,n,k,b)['num_gemm_ctas'])))]
print(f"\n  rgroup ladder  = {expect}  (graded rows 1..6)")
print(f"  rows that MOVE vs the previous ladder: {moved}")
if moved != [5]:
    print(f"  FATAL: expected exactly shape 5 to move, got {moved}"); ok = False
sys.exit(0 if ok else 1)
PY
rcA=$?
echo "  check A rc=$rcA"
[ $rcA -ne 0 ] && { echo "FATAL: geometry/rgroup check failed"; exit 1; }

echo
echo "=============== B. codegen reachability (PERSHAPE=0 must differ) ==============="
TMP=$D/logs/codegen_probe
mkdir -p "$TMP"
docker exec dhk-gemmrs bash -lc "
  hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
    -ffast-math --offload-arch=gfx942 -shared -fPIC \
    -I/home/subvadla/dhk/include -I/home/subvadla/dhk/include/pyutils -I$GEMM \
    -I\$ROCM_PATH/include/hip \
    -I/usr/local/lib/python3.12/dist-packages/pybind11/include -I/usr/include/python3.12 \
    -Wno-nan-infinity-disabled -ferror-limit=0 \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=0 \
    -DTK_MODNAME=gemm_rs_mi300x \
    $GEMM/gemm_rs_mi300x.cpp -o $TMP/ps0_same_modname.so" \
  > "$TMP/build.log" 2>&1
echo "  probe build rc=$? ($(stat -c%s "$TMP/ps0_same_modname.so" 2>/dev/null) bytes)"
SH=$(sha256sum "$SHIPPED" | cut -d' ' -f1)
P0=$(sha256sum "$TMP/ps0_same_modname.so" 2>/dev/null | cut -d' ' -f1)
echo "  shipped  (PERSHAPE=2) ${SH:0:16}  $(stat -c%s "$SHIPPED") B"
echo "  probe    (PERSHAPE=0) ${P0:0:16}  $(stat -c%s "$TMP/ps0_same_modname.so" 2>/dev/null) B"
if [ -z "$P0" ]; then echo "  WARN probe did not build; check B inconclusive"
elif [ "$SH" = "$P0" ]; then
  echo "  FATAL: identical -- PERSHAPE=2 is DEAD CODE, the change never reached codegen"
  exit 1
else
  echo "  ASSERT OK: objects differ, so PERSHAPE=2 changes generated code"
fi

echo
echo "=============== C. source default at measurement time ==============="
grep -n -A2 '^#ifndef HK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE' "$GEMM/gemm_rs_mi300x.cpp" | sed 's/^/  /'
grep -n 'HK_GEMM_RS_MI300X_RELEASE_GROUP ' "$GEMM/gemm_rs_mi300x.cpp" | head -4 | sed 's/^/  /'
grep -n 'HK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY' "$GEMM/gemm_rs_mi300x.cpp" | head -4 | sed 's/^/  /'
echo
echo "=============== D. ladder_mp.py rotation mode ==============="
grep -n 'rot_mode\|rot_rng\|LAD_ROT' "$D/ladder_mp.py" | sed 's/^/  /'
docker exec dhk-gemmrs python3 -m py_compile "$D/ladder_mp.py" && echo "  ladder_mp.py compiles"
echo "############ PREP COMPLETE ############"
