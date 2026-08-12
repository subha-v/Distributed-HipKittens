#!/usr/bin/env python3
"""Instrument B readout and the A-vs-B ordering cross-check.

Instrument B's purpose is NOT to reproduce A's absolutes -- it runs one process per
rank under the competition's own harness, which is a different topology and a
different timed region. Its job is to say whether that topology changes the ORDER of
the arms. If A and B disagree on ordering, A's one-process/8-device shortcut is
biasing the comparison and the headline is suspect.
"""
import json
import math
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
doc = json.load(open(os.path.join(D, "ladders.json")))
cc = doc.get("evaluator_crosscheck", {})
LB = doc["config"]["shapes"]

rows = {}
for rot, arms in sorted(cc.get("rotations", {}).items()):
    for name, p in sorted(arms.items()):
        tag = ("WARM(throwaway)" if p.get("is_warm")
               else "headline" if p.get("is_headline") else "?")
        rows[name] = (tag,
                      [s.get("best_us") if s else None for s in p.get("per_shape", [])],
                      p.get("geomean_us"), p.get("check"))

if not rows:
    print("NO instrument B rotations parsed. Raw tree:")
    for r, ds, fs in os.walk(os.path.join(D, "raw", "eval")):
        for f in fs:
            print("   ", os.path.join(r, f).replace(D + os.sep, ""))
    sys.exit(1)

hdr = f"{'arm/pass':<24}{'chk':>6}" + "".join(f"{l.split('x')[0]:>9}" for l in LB) + f"{'geomean':>10}"
print(hdr)
print("-" * len(hdr))
for name, (tag, best, gm, chk) in rows.items():
    line = f"{name:<24}{str(chk):>6}"
    for b in best:
        line += f"{b:>9.1f}" if b else f"{'--':>9}"
    line += f"{gm:>10.1f}" if gm else f"{'--':>10}"
    print(line + ("   " + tag if tag != "headline" else ""))


def gm_of(suffix):
    for k, v in rows.items():
        if k.endswith(suffix):
            return v[2]
    return None


o, r, k = gm_of("ours/benchmark"), gm_of("reference/benchmark"), gm_of("rank1/bench")
w = gm_of("rank1/warm")
A = doc["protocols"]["graded"]["arms"]
ao, ar, ak = (A["ours"]["geomean_us"], A["reference"]["geomean_us"],
              A["rank1"]["geomean_us"])

print("\n=== A vs B: absolutes, then ORDERING (the thing B is for) ===")
print(f"{'arm':<12}{'A graded':>11}{'B evaluator':>13}{'B/A':>8}")
for nm, a, b in (("ours", ao, o), ("reference", ar, r), ("rank1", ak, k)):
    if a and b:
        print(f"{nm:<12}{a:>11.1f}{b:>13.1f}{b/a:>7.2f}x")
if w and k:
    print(f"\nrank-1 warm pass {w:.1f} us vs bench {k:.1f} us "
          f"({w/k:.2f}x) -- warm is the JIT-filling throwaway, excluded")

if o and r and k:
    print(f"\nratios  A: ours/rank1 {ao/ak:.4f}   ours/reference {ao/ar:.4f}")
    print(f"        B: ours/rank1 {o/k:.4f}   ours/reference {o/r:.4f}")
    oa = [x[0] for x in sorted([("ours", ao), ("reference", ar), ("rank1", ak)],
                               key=lambda x: x[1])]
    ob = [x[0] for x in sorted([("ours", o), ("reference", r), ("rank1", k)],
                               key=lambda x: x[1])]
    print(f"\n        A order (fastest first): {oa}")
    print(f"        B order (fastest first): {ob}")
    print(f"        ORDERING {'AGREES' if oa == ob else 'DISAGREES'}")

    print("\n=== per-shape ordering agreement ===")
    for i in range(6):
        av = {n: A[n]["per_shape"][i]["best_us"] for n in ("ours", "reference", "rank1")
              if A[n]["per_shape"][i]}
        bv = {}
        for nm, suf in (("ours", "ours/benchmark"), ("reference", "reference/benchmark"),
                        ("rank1", "rank1/bench")):
            for kk, v in rows.items():
                if kk.endswith(suf) and v[1][i]:
                    bv[nm] = v[1][i]
        if len(av) == 3 and len(bv) == 3:
            sa = sorted(av, key=av.get)
            sb = sorted(bv, key=bv.get)
            print(f"  {LB[i]:>16}  A {str(sa):<40} B {str(sb):<40} "
                  f"{'agree' if sa == sb else 'DISAGREE'}")

print(f"\nparse refusals : {cc.get('parse_errors')}")
print(f"test-mode files skipped (expected, no timing blocks): "
      f"{len(cc.get('skipped_test_mode', []))}")
