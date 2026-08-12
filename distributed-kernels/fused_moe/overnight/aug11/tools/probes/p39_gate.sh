#!/usr/bin/env bash
# exp_24 RESOURCE + ISA GATE: unbundle, disassemble, and check every gate in one
# pass. Self-contained: never reads a stale .s.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
B=/opt/rocm/llvm/bin/clang-offload-bundler
O=/opt/rocm/llvm/bin/llvm-objdump
for t in base exp24; do
  rm -f "$OUTD/${t}.elf" "$OUTD/${t}.s"
  "$B" --type=o --unbundle --input="$OUTD/${t}.hsaco" \
       --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output="$OUTD/${t}.elf" 2>/dev/null
  "$O" -d --mcpu=gfx950 "$OUTD/${t}.elf" > "$OUTD/${t}.s"
  echo "$t: hsaco=$(stat -c %s "$OUTD/${t}.hsaco") elf=$(stat -c %s "$OUTD/${t}.elf") disasm_lines=$(wc -l < "$OUTD/${t}.s")"
done
echo
echo "=== resource tuple"
for t in base exp24; do
  echo "--- $t"
  grep -oE "(TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]|Occupancy \[waves/SIMD\]): [0-9]+" "$OUTD/${t}.log" | sort -u
done
echo
cat > /tmp/exp24_gate.py <<"PY"
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
cs = [i for i, x in enumerate(ins) if "cmpswap" in x]
print(f"--- {tag}")
print(f"  insns={len(ins)}  v_mfma={len(mf)}  scratch_ops={len(sc)}  "
      f"flat_atomic_pk_add_bf16={len(at)}  atomic_cmpswap={len(cs)}")
if not mf:
    print("  PARSE FAILED"); sys.exit(1)
spans, s, p = [], mf[0], mf[0]
for i in mf[1:]:
    if i - p <= 60: p = i
    else: spans.append((s, p)); s = p = i
spans.append((s, p))
big = sorted(((b - a + 1, a, b) for a, b in spans), reverse=True)
print(f"  MFMA spans={len(spans)} largest(len,first,last)={big[:2]}")
inside = [i for i in sc if any(a <= i <= b for a, b in spans)]
print(f"  GATE scratch ops INSIDE an MFMA span = {len(inside)}  (must be 0)")
if sc:
    print(f"  min |scratch - nearest mfma| = "
          f"{min(min(abs(i-j) for j in mf) for i in sc)} insns")
if at and sc:
    print(f"  GATE scratch ops within 24 insns of a pk_add_bf16 = "
          f"{sum(1 for i in sc if min(abs(i-j) for j in at) <= 24)}  (must be 0)")
for N in (4, 8, 16, 32):
    n = sum(1 for r in raw if r.startswith("s_waitcnt") and f"vmcnt({N})" in r)
    if n: print(f"  s_waitcnt vmcnt({N}) x{n}")
PY
for t in base exp24; do python3 /tmp/exp24_gate.py "$OUTD/${t}.s" "$t"; done
echo
echo "=== the depth compare on the SHIPPED path (exp24): scalar or vector?"
grep -n -B4 -A1 "vmcnt(8)" "$OUTD/exp24.s" | head -14
'
exit 0
