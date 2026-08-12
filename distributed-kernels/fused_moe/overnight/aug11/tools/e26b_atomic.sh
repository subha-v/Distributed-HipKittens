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
for tag in ("G0","G1"):
    rows = rows_of(tag + ".lisa")
    at = collections.Counter(r[1] for r in rows if "atomic" in r[1])
    print(tag, "atomic mnemonics:", dict(at))
    # the accumulate loop: sites attributed to phase2 lines 138..160
    sites = [i for i,r in enumerate(rows)
             if "atomic" in r[1] and r[3].startswith("n2_phase2_gm_mps.cpp:")
             and 130 <= int(r[3].split(":")[1]) <= 175]
    print(f"  atomics attributed to the accumulate loop (:130-175): {len(sites)}")
    nd = sum(1 for i in sites if any(rows[j][1]=="s_waitcnt" and "vmcnt(0)" in rows[j][2]
                                     for j in range(max(0,i-16), i)))
    ns = sum(1 for i in sites if any(rows[j][1].startswith("scratch_")
                                     for j in range(max(0,i-16), i)))
    print(f"  ...with a vmcnt(0) in the preceding 16 insns: {nd}")
    print(f"  ...with a scratch_* in the preceding 16 insns: {ns}")
PYEOF
'
