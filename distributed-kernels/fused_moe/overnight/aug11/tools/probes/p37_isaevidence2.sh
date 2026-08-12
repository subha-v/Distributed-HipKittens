#!/usr/bin/env bash
# exp_24 ISA evidence, parser fixed (objdump lines are "\t<mnemonic> ... // addr: hex").
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
cat > /tmp/exp24_map.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
pat = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-F]+):")
ins, raw = [], []
for ln in open(path, errors="ignore"):
    m = pat.match(ln)
    if m:
        ins.append(m.group(1)); raw.append(m.group(1) + " " + m.group(2).strip())
mf = [i for i, x in enumerate(ins) if x.startswith("v_mfma")]
sc = [i for i, x in enumerate(ins) if x.startswith("scratch_")]
at = [i for i, x in enumerate(ins) if x == "flat_atomic_pk_add_bf16"]
print(f"{tag}: insns={len(ins)} mfma={len(mf)} scratch={len(sc)} pk_add_bf16={len(at)}")
if not mf:
    print(f"{tag}: PARSE FAILED"); sys.exit(0)
spans, s, p = [], mf[0], mf[0]
for i in mf[1:]:
    if i - p <= 60: p = i
    else: spans.append((s, p)); s = p = i
spans.append((s, p))
big = sorted(((b - a + 1, a, b) for a, b in spans), reverse=True)
print(f"{tag}: MFMA spans={len(spans)}  largest (len,first,last)={big[:3]}")
inside = [i for i in sc if any(a <= i <= b for a, b in spans)]
print(f"{tag}: *** SCRATCH OPS INSIDE AN MFMA SPAN = {len(inside)} ***")
if sc:
    print(f"{tag}: min |scratch - nearest mfma| = "
          f"{min(min(abs(i-j) for j in mf) for i in sc)} insns")
for N in (4, 8, 16, 32):
    marks = [i for i, r in enumerate(raw)
             if r.startswith("s_waitcnt") and f"vmcnt({N})" in r]
    if marks:
        print(f"{tag}: s_waitcnt vmcnt({N}) x{len(marks)}  "
              f"min dist to mfma={min(min(abs(i-j) for j in mf) for i in marks)}")
# how many scratch ops sit within +/-24 insns of a pk_add_bf16 (the hot loop)
if at and sc:
    near = sum(1 for i in sc if min(abs(i-j) for j in at) <= 24)
    print(f"{tag}: scratch ops within 24 insns of a pk_add_bf16 = {near}")
# offset:68 localisation
o68 = [i for i, r in enumerate(raw) if r.startswith("scratch_") and "offset:68" in r]
if o68:
    d_at = [min(abs(i-j) for j in at) for i in o68] if at else []
    d_mf = [min(abs(i-j) for j in mf) for i in o68]
    print(f"{tag}: offset:68 ops={len(o68)}  min/median dist to pk_add="
          f"{min(d_at) if d_at else -1}/{sorted(d_at)[len(d_at)//2] if d_at else -1}"
          f"  min dist to mfma={min(d_mf)}")
    print(f"{tag}: offset:68 first 6 insn indices={o68[:6]}  total insns={len(ins)}")
PY
for t in base exp24; do python3 /tmp/exp24_map.py "$OUTD/${t}.s" "$t"; echo; done
echo "=== context around the first offset:68 scratch op in exp24"
grep -n -B10 -A4 "offset:68" "$OUTD/exp24.s" | head -34
'
exit 0
