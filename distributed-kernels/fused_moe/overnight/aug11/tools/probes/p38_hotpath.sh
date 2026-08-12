#!/usr/bin/env bash
# exp_24: is the SHIPPED (depth-8) epilogue loop free of scratch ops?
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
cat > /tmp/exp24_hot.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
pat = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-F]+):")
ins, raw = [], []
for ln in open(path, errors="ignore"):
    m = pat.match(ln)
    if m:
        ins.append(m.group(1)); raw.append(m.group(1) + " " + m.group(2).strip())
sc  = [i for i, x in enumerate(ins) if x.startswith("scratch_")]
at  = [i for i, x in enumerate(ins) if x == "flat_atomic_pk_add_bf16"]
def sites(N):
    return [i for i, r in enumerate(raw) if r.startswith("s_waitcnt") and f"vmcnt({N})" in r]
print(f"--- {tag}")
for N in (4, 8, 16, 32):
    s = sites(N)
    if not s: continue
    lo, hi = min(s), max(s)
    near = sum(1 for i in sc if lo - 40 <= i <= hi + 40)
    print(f"  vmcnt({N}): {len(s)} sites, insn range [{lo},{hi}], "
          f"scratch ops in range+/-40 = {near}")
# every scratch op: nearest pk_add distance, and nearest vmcnt-any distance
allthr = sorted(sum((sites(N) for N in (4,8,16,32)), []))
close = [i for i in sc if at and min(abs(i-j) for j in at) <= 24]
print(f"  scratch ops within 24 insns of a pk_add_bf16: {len(close)}")
for i in close[:8]:
    d_thr = min(abs(i-j) for j in allthr) if allthr else -1
    print(f"    idx={i} {raw[i][:60]!r} nearest_pk={min(abs(i-j) for j in at)} nearest_vmcntN={d_thr}")
# group the close ones by which region they belong to
if close:
    import statistics
    print(f"  close-op index range: {min(close)}..{max(close)} (total insns {len(ins)})")
    offs = {}
    for i in close:
        m = re.search(r"offset:(\d+)", raw[i]); k = m.group(1) if m else "none"
        offs[k] = offs.get(k, 0) + 1
    print(f"  close-op scratch offsets: {offs}")
PY
for t in base exp24; do python3 /tmp/exp24_hot.py "$OUTD/${t}.s" "$t"; done
echo
echo "=== context of one close scratch op (exp24): is it in the peer loop, the dual loop, or the donor loop?"
python3 - "$OUTD/exp24.s" <<"PY"
import re, sys
pat = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-F]+):")
ins, raw, lines = [], [], []
for ln in open(sys.argv[1], errors="ignore"):
    m = pat.match(ln)
    if m:
        ins.append(m.group(1)); raw.append(m.group(1) + " " + m.group(2).strip()); lines.append(ln.rstrip())
sc = [i for i, x in enumerate(ins) if x.startswith("scratch_")]
at = [i for i, x in enumerate(ins) if x == "flat_atomic_pk_add_bf16"]
close = [i for i in sc if min(abs(i-j) for j in at) <= 24]
if close:
    i = close[0]
    for j in range(max(0, i-18), min(len(lines), i+10)):
        mark = ">>" if j == i else "  "
        print(mark, j, lines[j][:100])
PY
'
exit 0
