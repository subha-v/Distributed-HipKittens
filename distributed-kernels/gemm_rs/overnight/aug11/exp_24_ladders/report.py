#!/usr/bin/env python3
"""exp_24 final report: read ladders.json and answer the dispatch's questions.

Everything printed here comes out of the artifact that will be plotted, so the
numbers reported upward and the numbers in the paper cannot drift apart.

Sections map 1:1 onto what was asked for:
  1  per-shape best AND median, all five arms, both protocols
  2  both geomeans
  3  graded vs pipelined ratio against rank-1 and reference, per shape + geomean,
     side by side, with the PROTOCOL DILUTION quantified
  4  the six null floors measured tonight, each beside its own shape's ratio
  5  whether the netted table survived
  6  correctness, integrity, and anything the parser refused
"""
import json
import math
import os
import statistics
import sys

D = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "ladders.json")
doc = json.load(open(path))

PUBLISHED = [1.34, 0.56, 0.61, 2.41, 2.17, 4.28]
AUG11 = [3.48, 0.93, 1.61, None, None, None]
PRIOR = doc["prior_denominator"]["ours_vs_rank1_per_shape"]
LABELS = doc["config"]["shapes"]
ARMS = ["ours", "ours_null", "reference", "rank1", "harness_floor"]


def geo(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(map(math.log, vals)) / len(vals)) if vals else None


def cell(proto, arm, i):
    try:
        return doc["protocols"][proto]["arms"][arm]["per_shape"][i]
    except (KeyError, IndexError):
        return None


print(f"################ exp_24 ladders.json ################")
print(f"file            : {os.path.basename(path)} "
      f"({os.path.getsize(path)} bytes)")
print(f"partial_run     : {doc['config']['partial_run']}")
print(f"shapes expected : {doc['config']['shapes_expected_instrument_a']}")
print(f"rotations exp'd : {doc['config']['rotations_expected_instrument_b']}")

print("\n################ 1-2. PER-SHAPE best / median, all arms, both protocols ################")
for proto in ("graded", "pipelined"):
    if proto not in doc["protocols"]:
        print(f"\n--- {proto}: ABSENT ---")
        continue
    arms = doc["protocols"][proto]["arms"]
    print(f"\n--- {proto.upper()} : best us  (median us in parentheses) ---")
    hdr = f"{'#':>2} {'shape':>16}" + "".join(f"{a:>22}" for a in ARMS if a in arms)
    print(hdr)
    print("-" * len(hdr))
    for i in range(6):
        line = f"{i+1:>2} {LABELS[i]:>16}"
        for a in ARMS:
            if a not in arms:
                continue
            c = cell(proto, a, i)
            line += (f"{c['best_us']:>12.2f} ({c['median_us']:>7.2f})"
                     if c else f"{'--':>22}")
        print(line)
    print("-" * len(hdr))
    for stat, key in (("geomean(best)", "geomean_us"),
                      ("geomean(median)", "geomean_median_us"),
                      ("geomean(mean)", "geomean_mean_us")):
        line = f"{stat:>19}"
        for a in ARMS:
            if a not in arms:
                continue
            v = arms[a].get(key)
            line += f"{v:>22.2f}" if v else f"{'--':>22}"
        print(line)

print("\n################ 3. RATIOS: graded vs pipelined, SIDE BY SIDE ################")
print("ours / rank-1 and ours / reference. >1 means the other arm is faster.")
print("`dilution` = pipelined ratio - graded ratio. Positive means the graded")
print("protocol's shared ~90 us constant FLATTERS us by compressing toward 1.")
for other in ("rank1", "reference"):
    for stat in ("best", "median"):
        g = doc["ratios"].get(f"ours_vs_{other}__graded__{stat}")
        p = doc["ratios"].get(f"ours_vs_{other}__pipelined__{stat}")
        if not g or not p:
            continue
        print(f"\n--- ours / {other}, statistic = {stat} ---")
        print(f"{'#':>2} {'shape':>16} {'graded':>9} {'pipelined':>10} "
              f"{'dilution':>9} {'floor%':>7} {'exp_14':>7}")
        dil = []
        for i in range(6):
            a = g["per_shape"][i]
            b = p["per_shape"][i]
            if not a or not b:
                print(f"{i+1:>2} {LABELS[i]:>16} {'--':>9} {'--':>10}")
                continue
            d = b["ratio"] - a["ratio"]
            dil.append(d)
            nf = doc["ratios"].get(f"null_floor__graded__best")
            fl = (abs(nf["per_shape"][i]["ratio"] - 1) * 100
                  if nf and nf["per_shape"][i] else float("nan"))
            prior = f"{PRIOR[i]:.3f}" if other == "rank1" else "-"
            print(f"{i+1:>2} {LABELS[i]:>16} {a['ratio']:>9.4f} "
                  f"{b['ratio']:>10.4f} {d:>+9.4f} {fl:>6.2f}% {prior:>7}")
        print(f"{'GEOMEAN':>19} {g['geomean']:>9.4f} {p['geomean']:>10.4f} "
              f"{p['geomean'] - g['geomean']:>+9.4f}")
        if dil:
            n_pos = sum(1 for d in dil if d > 0)
            print(f"  pipelined worse than graded on {n_pos}/{len(dil)} shapes; "
                  f"dilution median {statistics.median(dil):+.4f}, "
                  f"range [{min(dil):+.4f}, {max(dil):+.4f}]")
            print(f"  PATTERN {'HOLDS' if n_pos == len(dil) else 'DOES NOT HOLD'} "
                  f"across all shapes")

print("\n################ 4. THE SIX NULL FLOORS MEASURED TONIGHT ################")
print("ours vs ours_null: same file, two module names, same pool, same run.")
print("No ratio smaller than its own shape's floor is a result.")
print(f"{'#':>2} {'shape':>16} {'proto':>10} {'best%':>7} {'median%':>8} "
      f"{'mean%':>7} {'published':>10} {'aug11':>7} {'verdict':>26}")
floor_summary = {}
for proto in ("graded", "pipelined"):
    if proto not in doc["protocols"]:
        continue
    for i in range(6):
        a, b = cell(proto, "ours", i), cell(proto, "ours_null", i)
        if not a or not b:
            continue
        pct = {s: abs(a[f"{s}_us"] - b[f"{s}_us"]) / b[f"{s}_us"] * 100
               for s in ("best", "median", "mean")}
        floor_summary[(proto, i)] = pct
        ref = AUG11[i] if AUG11[i] else PUBLISHED[i]
        worst = max(pct["best"], pct["median"])
        verdict = ("within published" if worst <= PUBLISHED[i]
                   else "within aug11, over published" if worst <= (AUG11[i] or 1e9)
                   else "OVER both -- widen")
        print(f"{i+1:>2} {LABELS[i]:>16} {proto:>10} {pct['best']:>6.2f}% "
              f"{pct['median']:>7.2f}% {pct['mean']:>6.2f}% "
              f"{PUBLISHED[i]:>9.2f}% {(f'{AUG11[i]:.2f}%' if AUG11[i] else '-'):>7} "
              f"{verdict:>26}")

print("\n################ 5. DID THE NETTED TABLE SURVIVE? ################")
net = doc.get("graded_netted") or {}
print(f"floor source        : {net.get('floor_source')}")
print(f"floor per shape (us): {[round(v,2) if v else None for v in net.get('floor_per_shape_us',[])]}")
print(f"floor spread        : {net.get('floor_spread_pct')}")
print(f"shape-independent   : {net.get('floor_is_shape_independent')}")
for proto in ("graded",):
    hf = doc["protocols"].get(proto, {}).get("arms", {}).get("harness_floor")
    if hf:
        rsds = [r["rsd_pct"] for r in hf["per_shape"] if r]
        print(f"harness_floor rsd%  : "
              f"{[round(r,1) for r in rsds]}  (median {statistics.median(rsds):.1f}%)")
usable = net.get("floor_is_shape_independent") is True
print(f"\nVERDICT: netted table is {'USABLE' if usable else 'NOT USABLE -- DROP 2.2'}")
if usable:
    for other in ("rank1", "reference"):
        r = doc["ratios"].get(f"ours_vs_{other}__graded_netted__median")
        if r:
            per = [f"{x['ratio']:.3f}" if x else "--" for x in r["per_shape"]]
            print(f"  netted ours/{other:<10} geomean {r['geomean']:.4f}x  per-shape {per}")

print("\n################ 6. CORRECTNESS / INTEGRITY / REFUSALS ################")
bad = []
for shape, arms in doc.get("correctness", {}).items():
    for arm, c in arms.items():
        if not (c["allclose_1e-2"] and c["allclose_2e-3"]):
            bad.append((shape, arm, c))
print(f"gated arms failing either tolerance: {len(bad)}")
for shape, arm, c in bad:
    print(f"  FAIL {shape} {arm} {c}")
diffs = [c["max_abs_diff"] for arms in doc.get("correctness", {}).values()
         for c in arms.values()]
if diffs:
    print(f"max|diff| across all gated arms/shapes: "
          f"{min(diffs):.3e} .. {max(diffs):.3e}")

cc = doc.get("evaluator_crosscheck", {})
print(f"\ninstrument B rotations parsed : {sorted(cc.get('rotations', {}))}")
for rot, arms in sorted(cc.get("rotations", {}).items()):
    for name, p in sorted(arms.items()):
        tag = "WARM(throwaway)" if p.get("is_warm") else "headline"
        gm = p.get("geomean_us")
        print(f"  {rot} {name:<22} {tag:>16} geomean(best)="
              f"{gm:8.2f} us  check={p.get('check')}")
errs = cc.get("parse_errors", [])
print(f"\nPARSER REFUSALS: {len(errs)}")
for e in errs:
    print(f"  {e}")
print(f"test-mode files skipped (expected): {len(cc.get('skipped_test_mode', []))}")
