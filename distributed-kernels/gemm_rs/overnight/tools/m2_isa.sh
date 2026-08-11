#!/usr/bin/env bash
# Gate M2: per-instantiation resource tuple + ISA property checks.
#   - resource tuple from -Rpass-analysis=kernel-resource-usage
#   - ISA from --save-temps, then grep for the properties MI300X_VALIDATION.md
#     section "Gate M2" requires.
set -uo pipefail

REPO=/home/subvadla/dhk
SRC=$REPO/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
OUT=$REPO/distributed-kernels/gemm_rs/overnight/build
ISA=$OUT/isa
mkdir -p "$ISA"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

echo "################ resource tuples (from m1a.log) ################"
python3 - "$OUT/m1a.log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='replace').read().splitlines()
# Demangle the template args out of the mangled name: ILi32ELi256ELi32ELb0E
def pretty(mangled):
    nums = re.findall(r'Li(\d+)E', mangled)
    tail = re.search(r'Lb(\d)E', mangled)
    if len(nums) >= 3:
        return f"BM={nums[0]:>3} BN={nums[1]:>3} BK={nums[2]:>2} K_TAIL={tail.group(1) if tail else '?'}"
    return mangled

cur, rows = None, []
fields = {}
for line in log:
    m = re.search(r'Function Name:\s+(\S+)', line)
    if m:
        if cur: rows.append((cur, fields))
        cur, fields = pretty(m.group(1)), {}
        continue
    m = re.search(r'remark:\s+([A-Za-z][A-Za-z /]*?):\s+(\S+)', line)
    if m and cur:
        fields[m.group(1).strip()] = m.group(2)
if cur: rows.append((cur, fields))

keys = ["SGPRs","VGPRs","AGPRs","ScratchSize [bytes/lane]","Dynamic Stack",
        "Occupancy [waves/SIMD]","SGPRs Spill","VGPRs Spill","LDS Size [bytes/block]"]
hdr = f"{'instantiation':<34}" + "".join(f"{k.split(' [')[0][:9]:>10}" for k in keys)
print(hdr); print('-'*len(hdr))
for name, f in rows:
    print(f"{name:<34}" + "".join(f"{f.get(k,'-'):>10}" for k in keys))
print()
print("waves/SIMD -> CTAs/CU at 512 threads (8 waves/CTA, 4 SIMDs/CU):")
for name, f in rows:
    try:
        occ = float(f.get("Occupancy [waves/SIMD]", "0"))
        print(f"  {name:<34} {occ:>4} waves/SIMD = {occ*4/8:.2f} CTA/CU")
    except ValueError:
        pass
PY

echo
echo "################ generating ISA (--save-temps) ################"
cd "$ISA"
hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 -DTK_MODNAME=gemm_rs_mi300x \
  -I"$REPO/include" -I"$REPO/include/pyutils" -I"$ROCM_PATH/include/hip" \
  -I"$PBINC" -I"$PYINC" -Wno-nan-infinity-disabled \
  --save-temps -c "$SRC" -o "$ISA/gemm_rs_mi300x.o" >"$ISA/build.log" 2>&1
echo "save-temps exit=$?"
ls -la "$ISA" | grep -E '\.s$|\.o$' | head

S=$(ls "$ISA"/*gfx942*.s 2>/dev/null | head -1)
if [ -z "$S" ]; then S=$(ls "$ISA"/*.s 2>/dev/null | head -1); fi
echo "ISA file: $S"
[ -z "$S" ] && { echo "NO ISA FILE"; exit 1; }
wc -l "$S"

echo
echo "################ ISA property checks ################"
probe() { printf '%-46s %s\n' "$1" "$(grep -cE "$2" "$S")"; }
echo "-- emit path must lower to 16-byte stores --"
probe "global_store_dwordx4"        'global_store_dwordx4'
probe "flat_store_dwordx4"          'flat_store_dwordx4'
probe "buffer_store_dwordx4"        'buffer_store_dwordx4'
probe "global_store_dwordx2 (8B)"   'global_store_dwordx2'
probe "global_store_short (2B tail)" 'global_store_short'
echo "-- reduction load width --"
probe "global_load_dwordx4"         'global_load_dwordx4'
echo "-- cache/ordering ops --"
probe "buffer_wbl2  (writeback)"    'buffer_wbl2'
probe "buffer_inv   (invalidate)"   'buffer_inv'
probe "buffer_wbinvl1"              'buffer_wbinvl1'
probe "s_waitcnt vmcnt"             's_waitcnt +vmcnt'
probe "s_sleep (poll backoff)"      's_sleep'
probe "global_atomic"               'global_atomic'
probe "flat_atomic"                 'flat_atomic'
probe "s_barrier"                   's_barrier'
echo "-- MFMA --"
probe "v_mfma (any)"                'v_mfma'
probe "v_mfma_f32_16x16x16_bf16"    'v_mfma_f32_16x16x16.*bf16'
probe "v_mfma_f32_32x32x8"          'v_mfma_f32_32x32x8'
echo "-- spill indicators --"
probe "scratch_store"               'scratch_store'
probe "scratch_load"                'scratch_load'

echo
echo "-- per-kernel .s metadata (amdhsa) --"
grep -E '\.name:|\.sgpr_count|\.vgpr_count|\.agpr_count|\.group_segment_fixed_size|\.private_segment_fixed_size|\.max_flat_workgroup_size' "$S" | sed 's/^ *//' | head -80

echo
echo "################ DONE ################"
