#!/usr/bin/env python3
"""Read ladders_quick.json and answer the four questions the dispatch asked.

No GPU. Pure inspection of the artifact the quick run produced, so the numbers
reported upward come from the file that will be plotted rather than from a log.
"""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "ladders_quick.json")
doc = json.load(open(path))

# Published per-shape null floors, and aug11's re-measurement of shapes 1-3.
PUBLISHED = [1.34, 0.56, 0.61, 2.41, 2.17, 4.28]
AUG11 = [3.48, 0.93, 1.61, None, None, None]

print(f"=== {os.path.basename(path)} ===")
print(f"partial_run          : {doc['config']['partial_run']}")
print(f"shapes expected (A)  : {doc['config']['shapes_expected_instrument_a']}")
print(f"rotations expected(B): {doc['config']['rotations_expected_instrument_b']}")
print(f"caveats              : {len(doc['caveats'])}")

print("\n=== 1. schema keys present ===")
want_top = ["experiment", "node", "config", "protocols", "graded_netted",
            "correctness", "ratios", "evaluator_crosscheck",
            "prior_denominator", "caveats"]
missing = [k for k in want_top if k not in doc]
print(f"top-level: {'ALL PRESENT' if not missing else 'MISSING ' + str(missing)}")
for proto in ("graded", "pipelined"):
    arms = doc["protocols"][proto]["arms"]
    print(f"  {proto:>9}: arms={sorted(arms)}")
    row = next(r for r in arms["ours"]["per_shape"] if r)
    print(f"             per_shape row keys={sorted(row)}")
    print(f"             rows={len(arms['ours']['per_shape'])} "
          f"(must be 6, positionally aligned, null for absent)")
    print(f"             n samples/cell={row['n']}  "
          f"samples_us len={len(row['samples_us'])}")

print("\n=== 2. all five arms produced numbers, both protocols ===")
hdr = f"{'arm':>14} {'proto':>10} {'best':>9} {'median':>9} {'mean':>9} {'rsd%':>7} {'n':>6}"
print(hdr)
for proto in ("graded", "pipelined"):
    for arm, entry in doc["protocols"][proto]["arms"].items():
        for row in entry["per_shape"]:
            if not row:
                continue
            print(f"{arm:>14} {proto:>10} {row['best_us']:>9.2f} "
                  f"{row['median_us']:>9.2f} {row['mean_us']:>9.2f} "
                  f"{row['rsd_pct']:>7.1f} {row['n']:>6}")

print("\n=== 3. THE NULL-ARM FLOOR: ours vs ours_null, per shape ===")
print("Positive % = ours slower than its own twin. This is the ladder's own")
print("noise floor; no ratio smaller than it is a result.")
print(f"{'#':>2} {'shape':>16} {'stat':>7} {'proto':>10} {'ours':>9} "
      f"{'ours_null':>10} {'floor %':>9} {'published':>10} {'aug11':>7}")
floors = {}
for proto in ("graded", "pipelined"):
    arms = doc["protocols"][proto]["arms"]
    for i in range(6):
        a = arms["ours"]["per_shape"][i]
        b = arms["ours_null"]["per_shape"][i]
        if not a or not b:
            continue
        for stat in ("best_us", "median_us", "mean_us"):
            pct = abs(a[stat] - b[stat]) / b[stat] * 100.0
            floors[(proto, i, stat)] = pct
            pub = f"{PUBLISHED[i]:.2f}" if PUBLISHED[i] else "-"
            au = f"{AUG11[i]:.2f}" if AUG11[i] else "-"
            print(f"{i+1:>2} {a['shape']:>16} {stat.replace('_us',''):>7} "
                  f"{proto:>10} {a[stat]:>9.2f} {b[stat]:>10.2f} "
                  f"{pct:>8.2f}% {pub:>10} {au:>7}")

print("\n=== 4. harness_floor: is it measuring the protocol constant? ===")
for proto in ("graded", "pipelined"):
    for row in doc["protocols"][proto]["arms"]["harness_floor"]["per_shape"]:
        if not row:
            continue
        print(f"  {proto:>10}  best={row['best_us']:7.2f}  "
              f"median={row['median_us']:7.2f}  mean={row['mean_us']:7.2f}  "
              f"rsd={row['rsd_pct']:5.1f}%  n={row['n']}")
netted = doc.get("graded_netted") or {}
print(f"  floor_per_shape_us      : {netted.get('floor_per_shape_us')}")
print(f"  floor_spread_pct        : {netted.get('floor_spread_pct')}")
print(f"  floor_is_shape_indep    : {netted.get('floor_is_shape_independent')} "
      f"(null with one shape -- needs the full six)")
print("  expected band 57-99 us graded (57-79 empty region + 15-20 clone);")
print("  ~5-7 us pipelined (our host path against a 4.86 us floor)")

print("\n=== 5. ratios, both protocols, labelled ===")
for key in sorted(doc["ratios"]):
    entry = doc["ratios"][key]
    if entry and entry.get("geomean"):
        print(f"  {key:<46} {entry['geomean']:.4f}x")

print("\n=== 6. correctness gate ===")
for shape, arms in doc["correctness"].items():
    for arm, c in arms.items():
        print(f"  {shape:>16} {arm:>10}  1e-2={c['allclose_1e-2']}  "
              f"2e-3={c['allclose_2e-3']}  max|diff|={c['max_abs_diff']:.3e}")
