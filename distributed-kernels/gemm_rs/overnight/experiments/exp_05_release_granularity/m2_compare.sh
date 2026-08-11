#!/usr/bin/env bash
# CPU-only. Condition C3 asks for "zero new scratch in the M2 resource tuple",
# which needs a denominator: the same table for the frozen pre-E3 build. Both
# .so builds already emit -Rpass-analysis=kernel-resource-usage remarks into
# their own logs, so this just parses the two and prints them side by side.
set -uo pipefail

OUT=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/build

python3 - "$OUT/gemm_rs_mi300x_e3base.log" "$OUT/gemm_rs_mi300x.log" <<'PY'
import re, sys

FIELDS = [("VGPRs", "VGPR"), ("AGPRs", "AGPR"), ("TotalSGPRs", "SGPR"),
          ("ScratchSize \\[bytes/lane\\]", "scratch"),
          ("VGPRs Spill", "vspill"), ("SGPRs Spill", "sspill"),
          ("Occupancy \\[waves/SIMD\\]", "occ")]


def parse(path):
    text = open(path, errors="replace").read()
    out = {}
    for chunk in text.split("Function Name: ")[1:]:
        name = chunk.split("\n", 1)[0].strip()
        nums = re.findall(r"Li(\d+)E", name)
        tail = re.search(r"Lb(\d)E", name)
        key = (f"{nums[0]:>3}/{nums[1]:>3}/{nums[2]:>2} "
               f"tail={tail.group(1) if tail else '?'}") if len(nums) >= 3 else name
        row = {}
        for pattern, label in FIELDS:
            m = re.search(rf"{pattern}:\s*(\S+)", chunk)
            row[label] = m.group(1) if m else "-"
        out[key] = row
    return out


golden, cand = parse(sys.argv[1]), parse(sys.argv[2])
labels = [lab for _, lab in FIELDS]
head = f"{'BM/BN/BK':<18}" + "".join(f"{lab:>9}" for lab in labels)
print("pre-E3 golden vs E3 candidate; 'X -> Y' where they differ")
print(head)
print("-" * len(head))
regressed = []
for key in cand:
    g, c = golden.get(key, {}), cand[key]
    cells = ""
    for lab in labels:
        gv, cv = g.get(lab, "-"), c[lab]
        cells += f"{(cv if gv == cv else gv + '->' + cv):>9}"
    print(f"{key:<18}{cells}")
    if c["scratch"] != g.get("scratch") or c["vspill"] != g.get("vspill"):
        regressed.append(key)
print()
if regressed:
    print("C3 VIOLATED: new scratch / new VGPR spills on: " + ", ".join(regressed))
else:
    print("C3 OK: no new scratch and no new VGPR spills on any instantiation")
PY
