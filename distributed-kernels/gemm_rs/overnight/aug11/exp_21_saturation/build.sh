#!/usr/bin/env bash
# Build the exp_21 saturation ubench.
#
# Output goes to aug11/exp_21_saturation/build/, NEVER harness/build/ -- exp_20's ablation arms
# live there and a collision would corrupt another agent's experiment.
#
# Flag set copied from harness/build.sh. Note -I$ROCM_PATH/include/hip: without it <hip_bf16.h>
# is not found (the documented Gate-M1 command omits it; harness/build.sh has the working set).
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
EXP=$GEMM/overnight/aug11/exp_21_saturation
OUT=$EXP/build
mkdir -p "$OUT"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

COMMON=(
  -std=c++20 -O3
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math --offload-arch=gfx942
  -shared -fPIC
  -I"$REPO/include" -I"$REPO/include/pyutils"
  -I"$GEMM"
  -I"$ROCM_PATH/include/hip"
  -I"$PBINC" -I"$PYINC"
  -Wno-nan-infinity-disabled -ferror-limit=0
)

echo "########## building sat_ubench ##########"
hipcc "${COMMON[@]}" \
  -Rpass-analysis=kernel-resource-usage \
  "$EXP/sat_ubench.cpp" -o "$OUT/sat_ubench.so" 2>&1 | tee "$OUT/sat_ubench.log" | \
  grep -E 'error|Error' | head -40
status=${PIPESTATUS[0]}

echo
if [ -f "$OUT/sat_ubench.so" ] && [ "$status" = "0" ]; then
  echo "OK   sat_ubench.so ($(stat -c%s "$OUT/sat_ubench.so") bytes)"
else
  echo "FAIL sat_ubench (hipcc exit $status)"
  echo "---- first 60 diagnostic lines ----"
  grep -nE 'error|note:|required from' "$OUT/sat_ubench.log" | head -60
  exit 1
fi

echo
echo "########## resource tuples (per kernel) ##########"
python3 - "$OUT/sat_ubench.log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='replace').read().splitlines()

def pretty(mangled):
    if 'sat_kernel' not in mangled:
        return mangled.split('(')[0][:44]
    nums = re.findall(r'ILi(\d+)ELb(\d)ELb(\d)E', mangled)
    if nums:
        m, c, k = nums[0]
        return f"sat_kernel<mode={'abc'[int(m)]},conc={c},control={k}>"
    return mangled[:44]

cur, rows, fields = None, [], {}
for line in log:
    m = re.search(r'Function Name:\s+(\S+)', line)
    if m:
        if cur: rows.append((cur, fields))
        cur, fields = pretty(m.group(1)), {}
        continue
    m = re.search(r'remark:\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)', line)
    if m and cur:
        fields[m.group(1).strip()] = m.group(2)
if cur: rows.append((cur, fields))

keys = ["SGPRs","VGPRs","AGPRs","ScratchSize [bytes/lane]","Dynamic Stack",
        "Occupancy [waves/SIMD]","SGPRs Spill","VGPRs Spill","LDS Size [bytes/block]"]
hdr = f"{'kernel':<42}" + "".join(f"{k.split(' [')[0][:9]:>10}" for k in keys)
print(hdr); print('-'*len(hdr))
for name, f in rows:
    print(f"{name:<42}" + "".join(f"{f.get(k,'-'):>10}" for k in keys))
print()
print("waves/SIMD -> CTAs/CU at 512 threads (8 waves/CTA, 4 SIMDs/CU):")
for name, f in rows:
    try:
        occ = float(f.get("Occupancy [waves/SIMD]", "0"))
        print(f"  {name:<42} {occ:>4} waves/SIMD = {occ*4/8:.2f} CTA/CU")
    except ValueError:
        pass
PY

echo
echo "########## exported symbols ##########"
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('sat_ubench', '$OUT/sat_ubench.so')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('  entry points:', [x for x in dir(mod) if not x.startswith('_')])
print('  LDS_BYTES =', mod.LDS_BYTES, ' WIN_BYTES =', mod.WIN_BYTES,
      ' PACKETS_PER_WIN =', mod.PACKETS_PER_WIN, ' GRID_MAX =', mod.GRID_MAX)
" 2>&1 | tail -5

echo
echo "########## BUILD DONE ##########"
