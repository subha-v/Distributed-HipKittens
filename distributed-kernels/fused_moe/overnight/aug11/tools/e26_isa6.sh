#!/usr/bin/env bash
# exp_26 step 3g: final census detail + cuid confirmation.
set -uo pipefail
docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
echo "=== __hip_cuid_ symbols (explains the 41 .dynstr bytes) ==="
for v in B0 B1 B2; do printf "%s: " $v; strings $SC/out/$v.gfx950.elf | grep -o "__hip_cuid_[0-9a-f]*" | head -1; done
echo
echo "=== phase-1 K-loop instruction census by exact mnemonic (B1) ==="
awk -F"\t" "{print \$2}" $SC/out/B1.p1loop.txt | grep -E "^(ds_|v_mfma|buffer_|global_|flat_|scratch_|s_waitcnt|s_barrier)" | sort | uniq -c | sort -rn
echo
echo "=== same for B2 ==="
awk -F"\t" "{print \$2}" $SC/out/B2.p1loop.txt | grep -E "^(ds_|v_mfma|buffer_|global_|flat_|scratch_|s_waitcnt|s_barrier)" | sort | uniq -c | sort -rn
echo
echo "=== phase-2 K-loop VMEM check (hint says 8+7*kGM=29, census says?) ==="
python3 - <<PY
import re
def load(p):
    out=[];base=None
    for line in open(p,errors="replace"):
        m=re.match(r"^([0-9a-f]{16}) <",line)
        if m: base=int(m.group(1),16); continue
        m=re.match(r"^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):",line.rstrip())
        if m: out.append((int(m.group(3),16),m.group(1)))
    return out
ins=load("/home/subvadla/overnight-scratch/e26/out/B1.isa")
lo,hi=0x10b40,0x11c34
b=[m for a,m in ins if lo<=a<=hi]
import collections
c=collections.Counter()
for m in b:
    if m.startswith(("buffer_load","global_load","flat_load")): c["VMLD"]+=1
    elif m.startswith("ds_read"): c["DSR"]+=1
    elif m.startswith("ds_write"): c["DSW"]+=1
    elif m.startswith("v_mfma"): c["MFMA"]+=1
print("phase-2 K-loop per ITERATION (2 halves):",dict(c))
print("per HALF:",{k:v/2 for k,v in c.items()})
PY
'
echo "=== fetch vendor.diff ==="
scp -q -o BatchMode=yes subvadla@gbt350-odcdh2-c05-1.png-odc.dcgpu:/home/subvadla/overnight-scratch/e26/out/vendor.diff /dev/null 2>/dev/null || true
