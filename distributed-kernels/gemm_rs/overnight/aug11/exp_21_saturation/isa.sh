#!/usr/bin/env bash
# exp_21 ISA verification for the saturation ubench. CPU-only: hipcc --save-temps plus
# llvm-objdump on the module's own device code. Never touches a GPU, never touches
# harness/build/.
#
# What must be true, per kernel:
#   mode c  -> 16-byte peer stores (flat_store_dwordx4 or global_store_dwordx4), and NOT a
#              narrower store, and NOT buffer_store_* (whose voffset is 32-bit; that truncation
#              is the exact bug rank-1 hit, HANDOFF.md).
#   mode a  -> v_mfma_f32_16x16x16_bf16 and never v_mfma_f32_32x32x8, matching production.
#   mode b  -> 16-byte loads (global_load_dwordx4) for the 8 sources.
# Plus the resource tuple of every kernel.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
EXP=$GEMM/overnight/aug11/exp_21_saturation
OUT=$EXP/build
ISA=$OUT/isa
REPORT=$EXP/isa_report.txt
mkdir -p "$ISA"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

{
echo "################ exp_21 sat_ubench ISA report ################"
echo "date: $(date -Is)   host: $(hostname)"
echo "hipcc: $(hipcc --version 2>&1 | head -2 | tr '\n' ' ')"
echo

echo "################ generating ISA (--save-temps) ################"
cd "$ISA"
rm -f ./*.s ./*.o 2>/dev/null
hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -ffast-math --offload-arch=gfx942 \
  -I"$REPO/include" -I"$REPO/include/pyutils" -I"$GEMM" \
  -I"$ROCM_PATH/include/hip" -I"$PBINC" -I"$PYINC" \
  -Wno-nan-infinity-disabled -ferror-limit=0 \
  --save-temps -c "$EXP/sat_ubench.cpp" -o "$ISA/sat_ubench.o" >"$ISA/isa_build.log" 2>&1
echo "save-temps exit=$?"

S=$(ls "$ISA"/*gfx942*.s 2>/dev/null | head -1)
[ -z "$S" ] && S=$(ls "$ISA"/*.s 2>/dev/null | head -1)
echo "ISA file: $S"
if [ -z "$S" ]; then
  echo "NO ISA FILE -- last 40 lines of isa_build.log:"
  tail -40 "$ISA/isa_build.log"
  exit 1
fi
wc -l "$S"
echo

echo "################ whole-module opcode census ################"
probe() { printf '%-46s %s\n' "$1" "$(grep -cE "$2" "$S")"; }
echo "-- 16-byte stores (the peer-packet emit MUST be one of the first two) --"
probe "flat_store_dwordx4   (16 B, 64-bit flat addr)" 'flat_store_dwordx4'
probe "global_store_dwordx4 (16 B)"                   'global_store_dwordx4'
echo "-- narrower / suspect store forms --"
probe "global_store_dwordx3"       'global_store_dwordx3'
probe "global_store_dwordx2 (8 B)" 'global_store_dwordx2'
probe "global_store_dword   (4 B)" 'global_store_dword[^x]'
probe "global_store_short   (2 B)" 'global_store_short'
probe "buffer_store_ (32-bit voffset -- MUST be 0)" 'buffer_store_'
echo "-- 16-byte loads (the REDV=1 reduce and the LDS-staged emit source) --"
probe "global_load_dwordx4" 'global_load_dwordx4'
probe "flat_load_dwordx4"   'flat_load_dwordx4'
probe "ds_read_b128"        'ds_read_b128'
probe "ds_read2_b64"        'ds_read2_b64'
probe "ds_read_b64"         'ds_read_b64'
echo "-- MFMA --"
probe "v_mfma (any)"                 'v_mfma'
probe "v_mfma_f32_16x16x16_bf16"     'v_mfma_f32_16x16x16.*bf16'
probe "v_mfma_f32_32x32x8 (MUST be 0)" 'v_mfma_f32_32x32x8'
echo "-- ordering / cache ops (protocol arm only) --"
probe "buffer_wbl2 (release writeback)" 'buffer_wbl2'
probe "buffer_inv  (acquire invalidate)" 'buffer_inv'
probe "s_waitcnt vmcnt"                  's_waitcnt +vmcnt'
probe "global_atomic"                    'global_atomic'
probe "flat_atomic"                      'flat_atomic'
probe "s_barrier"                        's_barrier'
probe "s_memrealtime"                    's_memrealtime'
probe "s_setprio"                        's_setprio'
echo "-- spill indicators (MUST be 0 for a credible mode-a curve) --"
probe "scratch_store" 'scratch_store'
probe "scratch_load"  'scratch_load'
echo

echo "################ per-kernel section census ################"
# Split the .s by kernel symbol so each instantiation's opcodes are attributed to it.
python3 - "$S" <<'PY'
import re, sys
lines = open(sys.argv[1], errors='replace').read().splitlines()
starts = []
for i, l in enumerate(lines):
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_$.]*):\s*(;.*)?$', l)
    if m and ('sat_' in m.group(1) or 'kernel' in m.group(1)):
        starts.append((i, m.group(1)))
starts.append((len(lines), '<end>'))

def pretty(sym):
    if 'sat_kernel' in sym:
        m = re.search(r'ILi(\d)ELb(\d)ELb(\d)E', sym)
        if m:
            return f"sat_kernel<mode={'abc'[int(m.group(1))]},conc={m.group(2)},ctl={m.group(3)}>"
    for n in ('sat_calib_kernel', 'sat_fill_kernel', 'sat_verify_kernel'):
        if n in sym:
            return n
    return sym[:46]

pats = {
    'flat_store_dwordx4':  r'flat_store_dwordx4',
    'glob_store_dwordx4':  r'global_store_dwordx4',
    'store_dwordx2':       r'(flat|global)_store_dwordx2',
    'store_dword':         r'(flat|global)_store_dword[^x]',
    'store_short':         r'(flat|global)_store_short',
    'buffer_store':        r'buffer_store_',
    'glob_load_dwordx4':   r'global_load_dwordx4',
    'ds_read_b128':        r'ds_read_b128',
    'mfma_16x16x16_bf16':  r'v_mfma_f32_16x16x16.*bf16',
    'mfma_32x32x8':        r'v_mfma_f32_32x32x8',
    'buffer_wbl2':         r'buffer_wbl2',
    'atomic':              r'(global|flat)_atomic',
    's_barrier':           r's_barrier',
    'scratch':             r'scratch_(load|store)',
}
cols = list(pats)
print(f"{'kernel':<48}" + "".join(f"{c[:18]:>20}" for c in cols))
print('-' * (48 + 20 * len(cols)))
seen = set()
for (a, sym), (b, _) in zip(starts, starts[1:]):
    name = pretty(sym)
    if name in seen or name.startswith('.'):
        continue
    seen.add(name)
    body = '\n'.join(lines[a:b])
    counts = [len(re.findall(p, body)) for p in pats.values()]
    if sum(counts) == 0 and 'sat_' not in sym:
        continue
    print(f"{name:<48}" + "".join(f"{c:>20}" for c in counts))
PY
echo

echo "################ .s amdhsa metadata ################"
grep -E '\.name:|\.symbol:|\.sgpr_count|\.vgpr_count|\.agpr_count|\.group_segment_fixed_size|\.private_segment_fixed_size|\.max_flat_workgroup_size|\.sgpr_spill_count|\.vgpr_spill_count' "$S" \
  | sed 's/^ *//' | head -140
echo

echo "################ llvm-objdump cross-check on the built .so ################"
if [ -f "$OUT/sat_ubench.so" ]; then
  if command -v "$ROCM_PATH/llvm/bin/llvm-objdump" >/dev/null 2>&1; then
    OBJ="$ROCM_PATH/llvm/bin/llvm-objdump"
  else
    OBJ=llvm-objdump
  fi
  # roc-obj extracts the embedded device code objects; fall back to clang-offload-bundler.
  if command -v roc-obj >/dev/null 2>&1; then
    ( cd "$ISA" && roc-obj -o "$ISA/roc" "$OUT/sat_ubench.so" >"$ISA/rocobj.log" 2>&1 )
    echo "roc-obj exit=$? (log: $ISA/rocobj.log)"
    for f in "$ISA"/roc*gfx942*; do
      [ -f "$f" ] || continue
      echo "-- objdump $f --"
      "$OBJ" -d --mcpu=gfx942 "$f" > "$ISA/objdump.txt" 2>"$ISA/objdump.err"
      for p in flat_store_dwordx4 global_store_dwordx4 buffer_store_ v_mfma_f32_16x16x16 \
               v_mfma_f32_32x32x8 global_load_dwordx4 buffer_wbl2 scratch_store; do
        printf '   %-30s %s\n' "$p" "$(grep -c "$p" "$ISA/objdump.txt")"
      done
      break
    done
  else
    echo "roc-obj not present; the --save-temps .s census above is the record"
  fi
else
  echo "no sat_ubench.so yet -- run build.sh first"
fi

echo
echo "################ ISA REPORT DONE ################"
} 2>&1 | tee "$REPORT"
