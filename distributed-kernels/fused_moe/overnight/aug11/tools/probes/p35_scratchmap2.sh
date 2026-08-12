#!/usr/bin/env bash
# exp_24: scratch/MFMA locality, correct objdump line format this time.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
cat > /tmp/exp24_map.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
pat = re.compile(r"^\s*[0-9a-f]+:\t(\S+)(.*)$")
ins, raw = [], []
for ln in open(path, errors="ignore"):
    m = pat.match(ln)
    if m:
        ins.append(m.group(1)); raw.append(m.group(1) + " " + m.group(2).strip())
mf = [i for i, x in enumerate(ins) if x.startswith("v_mfma")]
sc = [i for i, x in enumerate(ins) if x.startswith("scratch_")]
print(f"{tag}: insns={len(ins)} mfma={len(mf)} scratch={len(sc)}")
if not mf: print(f"{tag}: PARSE FAILED"); sys.exit(0)
spans, s, p = [], mf[0], mf[0]
for i in mf[1:]:
    if i - p <= 60: p = i
    else: spans.append((s, p)); s = p = i
spans.append((s, p))
big = sorted(((b - a + 1, a, b) for a, b in spans), reverse=True)
print(f"{tag}: MFMA spans={len(spans)}; largest (len,first,last)={big[:4]}")
inside = [i for i in sc if any(a <= i <= b for a, b in spans)]
print(f"{tag}: scratch ops INSIDE an MFMA span = {len(inside)}")
if sc:
    print(f"{tag}: min |scratch - nearest mfma| = "
          f"{min(min(abs(i-j) for j in mf) for i in sc)} insns")
    lo, hi = min(mf), max(mf)
    print(f"{tag}: scratch before-first-mfma={sum(1 for i in sc if i<lo)} "
          f"between={sum(1 for i in sc if lo<=i<=hi)} "
          f"after-last-mfma={sum(1 for i in sc if i>hi)}")
for N in (4, 8, 16, 32):
    marks = [i for i, r in enumerate(raw) if r.startswith("s_waitcnt") and f"vmcnt({N})" in r]
    if not marks: continue
    first, last = marks[0], marks[-1]
    nsc = sum(1 for i in sc if first - 60 <= i <= last + 60)
    print(f"{tag}: vmcnt({N}) sites={len(marks)} region=[{first},{last}] "
          f"scratch ops in region+/-60 = {nsc}  "
          f"min dist to mfma = {min(min(abs(i-j) for j in mf) for i in marks)}")
PY
for t in base exp24; do python3 /tmp/exp24_map.py "$OUTD/${t}.s" "$t"; echo; done
echo "=== every atomic mnemonic"
for t in base exp24; do echo "--- $t"; grep -oE "\b(global|flat|buffer|ds)_atomic[a-z0-9_]*" "$OUTD/${t}.s" | sort | uniq -c; done
echo
echo "=== pk_add / bf16 atomic search"
for t in base exp24; do echo "--- $t"; grep -oE "\S*pk_add\S*" "$OUTD/${t}.s" | sort | uniq -c | head; done
echo
echo "=== scratch offsets touched"
for t in base exp24; do echo "--- $t"; grep -oE "scratch_[a-z_0-9]+ .*offset:[0-9]+" "$OUTD/${t}.s" | grep -oE "offset:[0-9]+" | sort -t: -k2 -n | uniq -c; done
'
exit 0
