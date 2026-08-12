#!/usr/bin/env bash
# exp_24: identify the value the epilogue is reloading from scratch.
set -uo pipefail
docker exec subha_k1 bash -lc '
set -uo pipefail
OUTD=/home/subvadla/exp24-stage/build
echo "=== readfirstlane present?"
grep -c "v_readfirstlane_b32" "$OUTD/exp24.s"
echo
python3 - "$OUTD/exp24.s" <<"PY"
import re, sys
pat = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-F]+):")
ins, raw, lines = [], [], []
for ln in open(sys.argv[1], errors="ignore"):
    m = pat.match(ln)
    if m:
        ins.append(m.group(1)); raw.append(m.group(1)+" "+m.group(2).strip()); lines.append(ln.rstrip())
sc = [i for i,x in enumerate(ins) if x.startswith("scratch_")]
at = [i for i,x in enumerate(ins) if x=="flat_atomic_pk_add_bf16"]
close = [i for i in sc if min(abs(i-j) for j in at) <= 24]
print("close scratch ops:", len(close))
print("their text (unique):")
for t in sorted({raw[i] for i in close}): print("   ", t)
i = close[1] if len(close) > 1 else close[0]
print()
print("=== context around close op #2 (idx", i, ")")
for j in range(max(0,i-8), min(len(lines), i+34)):
    print((">>" if j==i else "  "), j, lines[j][:105])
PY
'
exit 0
