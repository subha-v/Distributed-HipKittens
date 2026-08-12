#!/usr/bin/env bash
set -uo pipefail
docker exec subha_k1 bash -lc '
cd /home/subvadla/overnight-scratch/e26/out2
python3 - <<"PYEOF"
import re, collections
def rows_of(p):
    cur=None; out=[]
    for line in open(p, errors="replace"):
        m = re.match(r"^; (\S+):(\d+)", line)
        if m: cur = m.group(1).split("/")[-1]+":"+m.group(2); continue
        m = re.match(r"^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
        if m: out.append((int(m.group(3),16), m.group(1), m.group(2), cur or "-"))
    return out
AT = "flat_atomic_pk_add_bf16"
for tag in ("G0","G1"):
    rows = rows_of(tag + ".lisa")
    ats = [i for i,r in enumerate(rows) if r[1] == AT]
    print(f"=== {tag}: {len(ats)} {AT} sites ===")
    ns = sum(1 for i in ats if any(rows[j][1].startswith("scratch_") for j in range(max(0,i-20), i)))
    nd = sum(1 for i in ats if any(rows[j][1]=="s_waitcnt" and "vmcnt(0)" in rows[j][2]
                                   for j in range(max(0,i-20), i)))
    print(f"   with a scratch_* in the 20 insns before: {ns}")
    print(f"   with a vmcnt(0)  in the 20 insns before: {nd}")
    # distance from each migrated scratch site to the next atomic
    scr = [i for i,r in enumerate(rows) if r[1].startswith("scratch_") and r[3]=="n2_phase2_gm_mps.cpp:145"]
    d = []
    for i in scr:
        for j in range(i+1, min(i+40, len(rows))):
            if rows[j][1] == AT: d.append(j-i); break
    if d: print(f"   migrated-scratch -> next atomic distance: n={len(d)} min={min(d)} med={sorted(d)[len(d)//2]} max={max(d)}")
PYEOF
'
