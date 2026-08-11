#!/usr/bin/env bash
# Gate M2, part 2: exact resource tuples and in-context verification of the
# memory-ordering ISA claims (directional release, pure acquire, relaxed polls,
# device-cell epoch RMW).
set -uo pipefail

OUT=/home/subvadla/dhk/.node/build
S=$OUT/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s

echo "################ raw resource remarks ################"
grep -A11 'Function Name' "$OUT/m1a.log" | sed 's/.*remark: //; s/ \[-Rpass.*//' \
  | grep -vE '^--$|^/home' | sed 's/^ *//'

echo
echo "################ metadata table ################"
python3 - "$S" <<'PY'
import re, sys
txt = open(sys.argv[1], errors='replace').read()
# amdhsa.kernels metadata block at the end of the .s
blocks = re.findall(r'- \.agpr_count:.*?(?=\n  - \.agpr_count:|\namdhsa\.|\Z)', txt, re.S)
def g(b, key, default='-'):
    m = re.search(rf'\.{key}:\s*(\S+)', b)
    return m.group(1) if m else default
def pretty(n):
    nums = re.findall(r'Li(\d+)E', n); t = re.search(r'Lb(\d)E', n)
    return f"{nums[0]:>3}/{nums[1]:>3}/{nums[2]:>2} tail={t.group(1) if t else '?'}" if len(nums)>=3 else n
hdr = f"{'BM/BN/BK':<18}{'VGPR':>6}{'AGPR':>6}{'SGPR':>6}{'scratch':>9}{'staticLDS':>10}{'maxwg':>7}"
print(hdr); print('-'*len(hdr))
for b in blocks:
    print(f"{pretty(g(b,'name')):<18}{g(b,'vgpr_count'):>6}{g(b,'agpr_count'):>6}"
          f"{g(b,'sgpr_count'):>6}{g(b,'private_segment_fixed_size'):>9}"
          f"{g(b,'group_segment_fixed_size'):>10}{g(b,'max_flat_workgroup_size'):>7}")
print()
print("NOTE: LDS is requested dynamically at launch (extern __shared__), so")
print("      group_segment_fixed_size is 0 here; the real per-CTA LDS is")
print("      2*(BM+BN)*BK*2 bytes = 36864/32768/49152/65536/65536/24576/24576.")
PY

echo
echo "################ ordering ops in context ################"
echo "=== every buffer_wbl2 with 6 lines of context ==="
grep -n -B3 -A3 'buffer_wbl2' "$S" | head -80
echo
echo "=== every buffer_inv with 6 lines of context ==="
grep -n -B3 -A3 'buffer_inv' "$S" | head -80
echo
echo "=== s_sleep sites (bounded poll backoff) ==="
grep -n -B6 -A2 's_sleep' "$S" | head -60
echo
echo "=== atomic sites (expect epoch RMW + err atomicOr only) ==="
grep -n 'flat_atomic\|global_atomic\|buffer_atomic\|ds_add' "$S" | head -40
echo
echo "=== 16-byte peer stores ==="
grep -n 'global_store_dwordx4\|flat_store_dwordx4' "$S" | head -20
echo
echo "=== scratch (spill) sites ==="
grep -n -B4 -A2 'scratch_store\|scratch_load' "$S" | head -60
echo "################ DONE ################"
