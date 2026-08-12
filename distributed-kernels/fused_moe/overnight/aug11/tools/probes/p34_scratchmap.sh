#!/usr/bin/env bash
# exp_24: WHERE are the new scratch ops? Are any inside an MFMA K-loop span?
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
echo "=== objdump line format sample (base)"
grep -nE "v_mfma" "$OUTD/base.s" | head -3
echo "---"
grep -nE "scratch_load" "$OUTD/base.s" | head -3
echo
cat > /tmp/exp24_map.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
ins, addr = [], []
pat = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s)+\s*([a-z][a-z0-9_]*)")
for ln in open(path, errors="ignore"):
    m = pat.match(ln)
    if m:
        addr.append(int(m.group(1), 16)); ins.append(m.group(2))
mf = [i for i, x in enumerate(ins) if x.startswith("v_mfma")]
sc = [i for i, x in enumerate(ins) if x.startswith("scratch_")]
thr = [i for i, x in enumerate(ins) if x == "s_waitcnt"]
print(f"{tag}: parsed insns={len(ins)} mfma={len(mf)} scratch={len(sc)}")
if not mf:
    print(f"{tag}: PARSE FAILED"); sys.exit(0)
spans = []
s = p = mf[0]
for i in mf[1:]:
    if i - p <= 60: p = i
    else: spans.append((s, p)); s = p = i
spans.append((s, p))
spans_sorted = sorted(((b - a + 1, a, b) for a, b in spans), reverse=True)
print(f"{tag}: {len(spans)} MFMA spans; largest (insn_len, first, last): {spans_sorted[:4]}")
inside = [i for i in sc if any(a <= i <= b for a, b in spans)]
print(f"{tag}: scratch ops INSIDE an MFMA span = {len(inside)}")
if sc:
    print(f"{tag}: min |scratch - nearest mfma| = {min(min(abs(i-j) for j in mf) for i in sc)} insns")
    # bucket scratch ops by which epilogue-ish region they live in
    lo, hi = min(mf), max(mf)
    before = sum(1 for i in sc if i < lo)
    within = sum(1 for i in sc if lo <= i <= hi)
    after  = sum(1 for i in sc if i > hi)
    print(f"{tag}: scratch before-first-mfma={before} between-first-and-last-mfma={within} after-last-mfma={after}")
# distance from each throttle literal region to nearest mfma
PY
for t in base exp24; do python3 /tmp/exp24_map.py "$OUTD/${t}.s" "$t"; echo; done

echo "=== per-region scratch: window of +/-120 insns around each vmcnt(N) throttle literal"
cat > /tmp/exp24_thr.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
ins, raw = [], []
pat = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s)+\s*([a-z][a-z0-9_]*)(.*)$")
for ln in open(path, errors="ignore"):
    m = pat.match(ln)
    if m: ins.append(m.group(2)); raw.append(m.group(2) + " " + m.group(3).strip())
for N in (4, 8, 16, 32):
    marks = [i for i, r in enumerate(raw) if r.startswith("s_waitcnt") and f"vmcnt({N})" in r]
    if not marks: continue
    tot = 0
    for i in marks:
        lo, hi = max(0, i - 120), min(len(ins), i + 120)
        tot += sum(1 for j in range(lo, hi) if ins[j].startswith("scratch_"))
    print(f"{tag}: vmcnt({N}) sites={len(marks)} scratch ops within +/-120 insns (summed, overlapping) = {tot}")
PY
for t in base exp24; do python3 /tmp/exp24_thr.py "$OUTD/${t}.s" "$t"; echo; done

echo "=== which scratch offsets are loaded (exp24, top 12)"
grep -oE "scratch_load_[a-z0-9]+ +v[0-9\[\:\]]+, +off, +s[0-9]+ +offset:[0-9]+" "$OUTD/exp24.s" \
  | grep -oE "offset:[0-9]+" | sort | uniq -c | sort -rn | head -12
echo "=== base"
grep -oE "scratch_load_[a-z0-9]+ +v[0-9\[\:\]]+, +off, +s[0-9]+ +offset:[0-9]+" "$OUTD/base.s" \
  | grep -oE "offset:[0-9]+" | sort | uniq -c | sort -rn | head -12
echo
echo "=== atomic mnemonics present (both)"
for t in base exp24; do echo "--- $t"; grep -oE "global_atomic_[a-z0-9_]+" "$OUTD/${t}.s" | sort | uniq -c; done
'
exit 0
