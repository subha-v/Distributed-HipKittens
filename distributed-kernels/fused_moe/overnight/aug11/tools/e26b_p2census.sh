#!/usr/bin/env bash
# exp_26 follow-up step 8: (a) measured per-half VMEM census of PHASE 2's K-loop,
# to check its 0x020 hint (8 + 7*kGM = 29 at kGM=3) against reality;
# (b) the same accumulate-loop region in G0 (mask 0) vs G1 (mask 1).
set -uo pipefail
docker exec subha_k1 bash -lc '
cd /home/subvadla/overnight-scratch/e26/out2
python3 - <<"PYEOF"
import re, collections
exec(open("/home/subvadla/overnight-scratch/e26/py/mask.py").read().split("rows = []")[0])

print("=== per-half census of BOTH K-loops (D = donor phase1 + shipped phase2) ===")
for tag in ("D","M4"):
    insns = load(tag + ".isa")
    for name, nm in (("P1",96), ("P2",84)):
        lo,hi = find_loop(insns, nm)
        b = body(insns, lo, hi)
        c = collections.Counter(cls(x[1]) for x in b)
        print(f"  {tag} {name}: per-ITER  MFMA={c[chr(77)+chr(70)+chr(77)+chr(65)]} DSR={c[chr(68)+chr(83)+chr(82)]} "
              f"DSW={c[chr(68)+chr(83)+chr(87)]} VMLD={c[chr(86)+chr(77)+chr(76)+chr(68)]}   "
              f"per-HALF  MFMA={c[chr(77)+chr(70)+chr(77)+chr(65)]//2} DSR={c[chr(68)+chr(83)+chr(82)]/2} "
              f"DSW={c[chr(68)+chr(83)+chr(87)]/2} VMLD={c[chr(86)+chr(77)+chr(76)+chr(68)]/2}")

print()
print("=== accumulate-loop region: G0 (mask 0) vs G1 (mask 1) ===")
def rows_of(p):
    cur=None; out=[]
    for line in open(p, errors="replace"):
        m = re.match(r"^; (\S+):(\d+)", line)
        if m: cur = m.group(1).split("/")[-1]+":"+m.group(2); continue
        m = re.match(r"^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
        if m: out.append((int(m.group(3),16), m.group(1), m.group(2), cur or "-"))
    return out
for tag in ("G0","G1"):
    rows = rows_of(tag + ".lisa")
    # first remote atomic attributed to the accumulate loop
    ai = [i for i,r in enumerate(rows) if r[1].startswith("global_atomic")
          and r[3].startswith("n2_phase2_gm_mps.cpp:1")]
    print(f"--- {tag}: {len(ai)} global_atomic sites in the accumulate loop; first block ---")
    if not ai: continue
    i = ai[0]
    for j in range(i-14, i+2):
        a,m,o,s = rows[j]
        print("   %08X  %-24s %-40s [%s]" % (a, m, o[:40], s))
    # how many of the atomics have a vmcnt(0) within the 14 insns before them
    n = sum(1 for i in ai if any(rows[j][1]=="s_waitcnt" and "vmcnt(0)" in rows[j][2]
                                 for j in range(max(0,i-14), i)))
    print(f"   atomics with a vmcnt(0) drain in the preceding 14 insns: {n} / {len(ai)}")
PYEOF
'
