#!/usr/bin/env python3
"""exp_23 gate G4: scratch traffic strictly inside an MFMA span, per TU.

Span definition is the established one (aug11/tools/e34_42_gate1.sh): group the
v_mfma line indices of the disassembly into runs whose gaps are < 400 lines,
then count scratch_load / scratch_store strictly inside each run. A naive "any
scratch op after the first v_mfma" test is a DIFFERENT and wrong check -- it
fires on the reference arm itself, because the kernel's two MFMA spans are
separated by ordinary spill-carrying code that legitimately touches scratch.

Usage: e23_g4_spans.py LABEL=path.isa [LABEL=path.isa ...]
Exit 1 if any TU has a scratch op inside a span.
"""
import re
import sys

SCRATCH = re.compile(r"scratch_(load|store)")
bad_tus = []

for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    lines = open(path).readlines()
    idx = [i for i, l in enumerate(lines) if "v_mfma" in l]
    if not idx:
        print("%-5s NO v_mfma FOUND -- wrong ISA file?" % label)
        bad_tus.append(label)
        continue
    spans, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 400:
            spans.append((start, prev))
            start = i
        prev = i
    spans.append((start, prev))

    total = 0
    print("---- %s ----" % label)
    for a, b in spans:
        n = sum(1 for l in lines[a:b + 1] if SCRATCH.search(l))
        m = sum(1 for j in idx if a <= j <= b)
        total += n
        print("  span %6d-%-6d  %3d mfma  scratch ops inside = %d" % (a, b, m, n))
    print("  mfma total %d in %d span(s); scratch ops inside spans = %d  %s"
          % (len(idx), len(spans), total, "PASS" if total == 0 else "** FAIL **"))
    if total:
        bad_tus.append(label)

print("G4 %s" % ("PASS on every TU" if not bad_tus
                 else "** FAIL ** on " + ",".join(bad_tus)))
sys.exit(1 if bad_tus else 0)
