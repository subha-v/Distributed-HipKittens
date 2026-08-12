#!/usr/bin/env bash
# exp_26 follow-up step 7: what is n2_phase2_gm_mps.cpp:145 IN THE SNAPSHOT, and
# what instruction sits next to each of the 96 migrated scratch loads?
set -uo pipefail
SC=$HOME/overnight-scratch/e26
F=$SC/dhk/distributed-kernels/fused_moe/n2_phase2_gm_mps.cpp

echo "===SNAPSHOT n2_phase2_gm_mps.cpp:138..160==="
sed -n '138,160p' "$F"
echo "===SNAPSHOT FILE ID==="
sha256sum "$F"; wc -l "$F"

echo "===WHAT FOLLOWS EACH MIGRATED scratch_load IN G1==="
docker exec subha_k1 bash -lc '
cd /home/subvadla/overnight-scratch/e26/out2
python3 - <<"PYEOF"
import re, collections
cur=None; rows=[]
for line in open("G1.lisa", errors="replace"):
    m = re.match(r"^; (\S+):(\d+)", line)
    if m: cur=(m.group(1).split("/")[-1], int(m.group(2))); continue
    m = re.match(r"^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
    if m: rows.append((int(m.group(3),16), m.group(1), m.group(2), cur))
idx=[i for i,r in enumerate(rows) if r[1].startswith("scratch_") and r[3]==("n2_phase2_gm_mps.cpp",145)]
print(f"migrated scratch sites: {len(idx)}")
nxt=collections.Counter()
for i in idx:
    for j in range(i+1, min(i+9, len(rows))):
        if rows[j][1].startswith(("global_atomic","buffer_atomic","flat_atomic","global_store","buffer_store")):
            nxt[(rows[j][1], j-i)] += 1; break
for k,v in nxt.most_common(10): print("   next memory op:", k, "count", v)
print()
print("--- one site verbatim ---")
i=idx[0]
for j in range(i-2, i+8):
    a,m,o,s = rows[j]
    print(f"  {a:08X}  {m:24s} {o[:60]:60s}  [{s[0] if s else '-'}:{s[1] if s else '-'}]")
PYEOF
'
