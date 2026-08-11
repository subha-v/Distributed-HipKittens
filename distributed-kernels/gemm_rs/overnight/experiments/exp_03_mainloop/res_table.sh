#!/usr/bin/env bash
# Per-instantiation resource tuple for one arm, from BOTH sources:
#   (a) -Rpass-analysis=kernel-resource-usage remarks (VGPR/AGPR/SGPR/spills/occupancy)
#   (b) the amdhsa YAML in the .s (vgpr_count / agpr_count / private_segment_fixed_size)
# arm dir defaults to the last one built.
set -uo pipefail
ARM=${1:?usage: res_table.sh <arm_name>}
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
A=$ON/experiments/exp_03_mainloop/arms/$ARM

python3 - "$A/m1_build.log" "$A/gemm_rs_mi300x.gfx942.s" <<'PY'
import re, sys
log = open(sys.argv[1], errors='replace').read().splitlines()

def pretty(m):
    n = re.findall(r'Li(\d+)E', m); t = re.search(r'Lb(\d)E', m)
    return f"{n[0]}/{n[1]}/{n[2]} tail={t.group(1) if t else '?'}" if len(n) >= 3 else m

cur, rows, f = None, [], {}
for line in log:
    m = re.search(r'Function Name:\s+(\S+)', line)
    if m:
        if cur: rows.append((cur, f))
        cur, f = pretty(m.group(1)), {}
        continue
    m = re.search(r'remark:\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)', line)
    if m and cur: f[m.group(1).strip()] = m.group(2)
if cur: rows.append((cur, f))

keys = [("SGPRs","SGPR"),("VGPRs","VGPR"),("AGPRs","AGPR"),
        ("ScratchSize [bytes/lane]","scratch"),("SGPRs Spill","sSpill"),
        ("VGPRs Spill","vSpill"),("Occupancy [waves/SIMD]","w/SIMD")]
hdr = f"{'BM/BN/BK':<20}" + "".join(f"{s:>9}" for _, s in keys)
print("############ -Rpass kernel-resource-usage ############")
print(hdr); print('-' * len(hdr))
for name, fl in rows:
    print(f"{name:<20}" + "".join(f"{fl.get(k,'-'):>9}" for k, _ in keys))

txt = open(sys.argv[2], errors='replace').read()
blocks = re.findall(r'- \.agpr_count:.*?(?=\n  - \.agpr_count:|\namdhsa\.|\Z)', txt, re.S)
def g(b, key, d='-'):
    m = re.search(rf'\.{key}:\s*(\S+)', b)
    return m.group(1) if m else d
print()
print("############ amdhsa metadata in the .s ############")
hdr2 = f"{'BM/BN/BK':<20}{'VGPR':>7}{'AGPR':>7}{'SGPR':>7}{'scratch':>9}"
print(hdr2); print('-' * len(hdr2))
for b in blocks:
    print(f"{pretty(g(b,'name')):<20}{g(b,'vgpr_count'):>7}{g(b,'agpr_count'):>7}"
          f"{g(b,'sgpr_count'):>7}{g(b,'private_segment_fixed_size'):>9}")
PY
