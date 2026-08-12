#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape

echo "########## round-2 resource tuples (does ps2 keep ps0's register budget?) ##########"
sed -n '/arm resource tuples/,/sha256/p' "$D/logs/build_arms2.log"
echo
sed -n '/sha256/,/ARM MODULES/p' "$D/logs/build_arms2.log"

echo
echo "########## round-2 forward pass ##########"
sed -n '/PER-SHAPE SUMMARY/,$p' "$D/logs/ab2_fwd.log" 2>/dev/null | head -40

echo
echo "########## ALL PASSES POOLED ##########"
python3 - "$D/logs" <<'PY'
import json, os, statistics, sys, math
logs = sys.argv[1]
files = [("r1_fwd", "ab_pershape_ab_fwd.json"), ("r1_rev", "ab_pershape_ab_rev.json"),
         ("r2_fwd", "ab_pershape_ab2_fwd.json"), ("r2_rev", "ab_pershape_ab2_rev.json")]
runs = []
for tag, f in files:
    p = os.path.join(logs, f)
    if os.path.exists(p):
        runs.append((tag, json.load(open(p))))
print("passes:", ", ".join(t for t, _ in runs))
arms = ["ps0", "ps0b", "ps1", "ps2", "rg2c"]
keys = list(runs[0][1]["per_shape"])

print("\n--- per-pass BEST, shape 5 (8192x4096x14336), the only row that changes ---")
k5 = keys[4]
print(f"{'pass':<8}" + "".join(f"{a:>10}" for a in arms) + "   fastest arm")
for tag, r in runs:
    row = r["per_shape"][k5]["arms"]
    vals = {a: min(row[a]["samples"]) for a in arms if a in row}
    line = f"{tag:<8}" + "".join(
        f"{vals[a]:>10.1f}" if a in vals else f"{'-':>10}" for a in arms)
    print(line + f"   {min(vals, key=vals.get)}")

print("\n--- ps0 vs ps0b: TWO BUILDS OF THE SAME RULE, per pass, per shape (%) ---")
print(f"{'#':<3}{'shape':<20}" + "".join(f"{t:>10}" for t, _ in runs))
for i, k in enumerate(keys):
    line = f"{i+1:<3}{k:<20}"
    for tag, r in runs:
        a = r["per_shape"][k]["arms"]
        d = min(a["ps0b"]["samples"]) / min(a["ps0"]["samples"]) - 1
        line += f"{d*100:>+9.2f}%"
    print(line)

print("\n--- every arm vs ps0, POOLED over all passes (best of pooled samples) ---")
print(f"{'#':<3}{'shape':<20}{'t/CTA':>6}" + "".join(f"{a:>11}" for a in arms))
pooled_all = {}
for i, k in enumerate(keys):
    pooled = {}
    for a in arms:
        s = []
        for _, r in runs:
            if a in r["per_shape"][k]["arms"]:
                s += r["per_shape"][k]["arms"][a]["samples"]
        if s:
            pooled[a] = s
    pooled_all[k] = pooled
    ppc = runs[0][1]["per_shape"][k]["arms"]["_geometry"]["tiles_per_cta"]
    base = min(pooled["ps0"])
    line = f"{i+1:<3}{k:<20}{ppc:>6}"
    for a in arms:
        line += (f"{min(pooled[a]):>7.1f}"
                 f"{(min(pooled[a])/base-1)*100:>+5.1f}") if a in pooled else f"{'-':>11}"
    print(line)

print("\n--- separation test: is any arm's pooled range disjoint from ps0's? ---")
for i, k in enumerate(keys):
    p = pooled_all[k]
    lo0, hi0 = min(p["ps0"]), max(p["ps0"])
    out = []
    for a in arms[1:]:
        if a not in p:
            continue
        lo, hi = min(p[a]), max(p[a])
        out.append(f"{a}:{'DISJOINT' if (hi < lo0 or hi0 < lo) else 'overlap'}")
    print(f"{i+1}: {k:<20} ps0 [{lo0:.1f},{hi0:.1f}]   " + "  ".join(out))
PY
