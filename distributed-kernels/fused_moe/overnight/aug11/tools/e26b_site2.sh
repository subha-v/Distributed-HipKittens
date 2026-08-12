#!/usr/bin/env bash
set -uo pipefail
docker exec subha_k1 bash -lc '
cd /home/subvadla/overnight-scratch/e26/out2
python3 - <<"PYEOF"
import re, collections
cur=None; rows=[]
for line in open("G1.lisa", errors="replace"):
    m = re.match(r"^; (\S+):(\d+)", line)
    if m:
        cur = m.group(1).split("/")[-1] + ":" + m.group(2); continue
    m = re.match(r"^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
    if m: rows.append((int(m.group(3),16), m.group(1), m.group(2), cur or "-"))
tgt = "n2_phase2_gm_mps.cpp:145"
idx = [i for i,r in enumerate(rows) if r[1].startswith("scratch_") and r[3] == tgt]
print("migrated scratch sites:", len(idx))
nxt = collections.Counter()
for i in idx:
    for j in range(i+1, min(i+12, len(rows))):
        if rows[j][1].startswith(("global_atomic","buffer_atomic","flat_atomic","global_store","buffer_store","ds_bpermute")):
            nxt[rows[j][1] + " at +" + str(j-i)] += 1
            break
for k,v in nxt.most_common(10):
    print("   followed by", k, "->", v, "sites")
print()
print("--- one site verbatim ---")
i = idx[0]
for j in range(i-2, i+9):
    a,m,o,s = rows[j]
    print("  %08X  %-26s %-52s [%s]" % (a, m, o[:52], s))
PYEOF
'
