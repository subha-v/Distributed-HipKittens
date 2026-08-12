#!/usr/bin/env bash
# Pull every arm's M7 rotation samples out of the archived ladder runs and print
# best / median / mean per shape, plus the single-shot wall the review's C8 asks
# for. m7_bench reports the mean of its rotations; on a node with a ~1.3%
# run-to-run floor the ledger's convention is best and median, so those are
# recomputed here from the samples the ladder already recorded.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_05_release_granularity

python3 - "$D" <<'PY'
import json, math, os, statistics, sys

D = sys.argv[1]
ARMS = ["N1", "N2", "N4", "N4c"]
SHAPES = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
          "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def load(arm):
    p = os.path.join(D, "arms", arm, "m7_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as h:
        return json.load(h)


rows = {}
for arm in ARMS:
    d = load(arm)
    if d is None:
        print(f"(no m7_results.json for {arm})")
        continue
    s = d["samples_us"]
    rows[arm] = {i: sorted(s[str(i)]) for i in range(6)}

print("M7 pipelined wall, us. Each cell is best / median over the 3 rotations.")
print(f"{'shape':<20}" + "".join(f"{a:>20}" for a in rows))
print("-" * (20 + 20 * len(rows)))
for i, name in enumerate(SHAPES):
    line = f"{name:<20}"
    for arm in rows:
        v = rows[arm][i]
        line += f"{min(v):>10.2f}/{statistics.median(v):<9.2f}"
    print(line)
print("-" * (20 + 20 * len(rows)))
line = f"{'GEOMEAN':<20}"
for arm in rows:
    b = geomean([min(rows[arm][i]) for i in range(6)])
    m = geomean([statistics.median(rows[arm][i]) for i in range(6)])
    line += f"{b:>10.2f}/{m:<9.2f}"
print(line)

base = "N1"
if base in rows:
    print(f"\nratio of best vs {base}")
    for arm in rows:
        r = [min(rows[arm][i]) / min(rows[base][i]) for i in range(6)]
        gb = (geomean([min(rows[arm][i]) for i in range(6)])
              / geomean([min(rows[base][i]) for i in range(6)]))
        print(f"  {arm:<5}" + "".join(f"{x:>8.4f}" for x in r)
              + f"   geomean {gb:.4f}")

print("\nsingle-shot wall (C8), us, from each arm's m7 log rotation 0")
for arm in ARMS:
    p = os.path.join(D, "arms", arm, "m7_bench.log")
    if not os.path.exists(p):
        continue
    vals = []
    for line in open(p):
        if "single_wall=" in line and len(vals) < 6:
            vals.append(float(line.split("single_wall=")[1].split("us")[0]))
    print(f"  {arm:<5}" + "".join(f"{v:>10.2f}" for v in vals))
PY
