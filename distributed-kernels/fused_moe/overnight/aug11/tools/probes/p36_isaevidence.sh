#!/usr/bin/env bash
# exp_24 final ISA evidence: unbundle, disassemble, and check every gate.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
B=/opt/rocm/llvm/bin/clang-offload-bundler
O=/opt/rocm/llvm/bin/llvm-objdump
for t in base exp24; do
  "$B" --type=o --unbundle --input="$OUTD/${t}.hsaco" \
       --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$OUTD/${t}.elf" 2>/dev/null
  "$O" -d --mcpu=gfx950 "$OUTD/${t}.elf" > "$OUTD/${t}.s"
done

cat > /tmp/exp24_map.py <<"PY"
import re, sys
path, tag = sys.argv[1], sys.argv[2]
pat = re.compile(r"^\s*\d+:\t(\S+)(.*)$")
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
print(f"{tag}: SCRATCH OPS INSIDE AN MFMA SPAN = {len(inside)}")
if sc:
    print(f"{tag}: min |scratch - nearest mfma| = "
          f"{min(min(abs(i-j) for j in mf) for i in sc)} insns")
for N in (4, 8, 16, 32):
    marks = [i for i, r in enumerate(raw)
             if r.startswith("s_waitcnt") and f"vmcnt({N})" in r]
    if marks:
        print(f"{tag}: s_waitcnt vmcnt({N}) x{len(marks)}  "
              f"min dist to mfma={min(min(abs(i-j) for j in mf) for i in marks)}")
PY
for t in base exp24; do python3 /tmp/exp24_map.py "$OUTD/${t}.s" "$t"; echo; done

echo "=== scratch offsets touched (count per offset)"
for t in base exp24; do
  echo "--- $t"
  grep -oE "scratch_[a-z_0-9]+ .*offset:[0-9]+" "$OUTD/${t}.s" \
    | grep -oE "offset:[0-9]+" | sort -t: -k2 -n | uniq -c
done

echo
echo "=== all atomic mnemonics"
for t in base exp24; do
  echo "--- $t"
  grep -oE "\b(flat|global|buffer|ds)_atomic[a-z0-9_]*" "$OUTD/${t}.s" | sort | uniq -c
done

echo
echo "=== the shipped depth-8 site, in context (exp24, first occurrence)"
grep -n -B6 -A6 "vmcnt(8)" "$OUTD/exp24.s" | head -30
'
exit 0
