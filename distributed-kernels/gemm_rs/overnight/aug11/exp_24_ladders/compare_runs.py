#!/usr/bin/env python3
"""exp_24 re-measure: the paired comparison against the previous ladder.

Answers the four questions the dispatch asks of the numbers:

  1. per-shape delta, new (fb3d670b, exp_26 shipped) vs previous (79599cce,
     pre-exp_26), marked resolved/unresolved against that shape's OWN null floor
     measured in this run;
  2. did the pre-registration hold -- shape 5 improves, the other five flat? A
     resolvable move on a shape whose rgroup did not change is a blocker;
  3. the new ours/rank-1 and ours/reference ratios, and whether the geomean
     moved by more than the 2% ratio-noise-floor rule allows us to call a change;
  4. did the null floors change? This run also switched the arm ordering from
     the defective cyclic scheme to a per-rep shuffle, and the null arm is the
     detector for whether that mattered.

Resolution rule: a delta counts only if |delta| exceeds the larger of the two
runs' floors on that shape and statistic. Using the larger is the conservative
choice -- the floor is an estimate too, and this run's floor is measured under a
different ordering scheme than the previous run's.
"""
import json
import math
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(D, "ladders.json")
PREV = os.path.join(D, "prev_79599cce", "ladders.json")
# rgroup ladder before -> after. Only shape 5 (index 4) changes; every other row
# runs a bit-identical instruction stream, so those five ARE the control.
RGROUP_PREV = [1, 1, 1, 1, 1, 4]
RGROUP_NOW = [1, 1, 1, 1, 2, 4]
CHANGED = [i for i in range(6) if RGROUP_PREV[i] != RGROUP_NOW[i]]
# exp_26 measured shape 5 at -6.56% pipelined and projected the graded geomean
# to ~1.085x. Both are pre-registrations to be scored, not inputs.
EXP26_SHAPE5_PIPELINED_PCT = -6.56
EXP26_PROJECTED_GRADED_GEOMEAN_RATIO = 1.085
RATIO_NOISE_FLOOR_PCT = 2.0


def geo(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def cell(doc, proto, arm, i):
    try:
        return doc["protocols"][proto]["arms"][arm]["per_shape"][i]
    except (KeyError, IndexError, TypeError):
        return None


def floor_pct(doc, proto, i, stat):
    """|ours - ours_null| / ours_null on one shape: the same definition report.py
    uses, so the numbers are comparable to the published 0.34/0.44/0.48/2.15/
    2.00/4.31% set."""
    a, b = cell(doc, proto, "ours", i), cell(doc, proto, "ours_null", i)
    if not a or not b:
        return None
    return abs(a[f"{stat}_us"] - b[f"{stat}_us"]) / b[f"{stat}_us"] * 100.0


def main():
    if not os.path.exists(PREV):
        print(f"FATAL: no previous ladder at {PREV}")
        return 1
    new = json.load(open(NEW))
    prev = json.load(open(PREV))
    labels = new["config"]["shapes"]
    print("################ module identity ################")
    for tag, doc in (("previous", prev), ("new", new)):
        sha = (doc.get("node", {}).get("module_sha256")
               or doc.get("node", {}).get("modules") or "unrecorded")
        print(f"  {tag:>8}: {sha}")
    print(f"  rgroup prev {RGROUP_PREV}  ->  now {RGROUP_NOW}")
    print(f"  rows whose codegen changed: {[i + 1 for i in CHANGED]}")
    print(f"  arm ordering: prev={prev['config'].get('rot_mode', 'cyclic')}  "
          f"new={new['config'].get('rot_mode', 'shuffle')}")

    blockers, verdicts = [], {}
    for proto in ("graded", "pipelined"):
        for stat in ("best", "median"):
            print(f"\n################ delta, {proto} / {stat} ################")
            print(f"{'#':>2} {'shape':>16} {'rg':>5} {'prev us':>10} "
                  f"{'new us':>10} {'delta%':>8} {'floor%':>7} {'verdict':>22}")
            for i in range(6):
                a, b = cell(prev, proto, "ours", i), cell(new, proto, "ours", i)
                if not a or not b:
                    print(f"{i+1:>2} {labels[i]:>16} {'':>5} "
                          f"{'MISSING':>10}")
                    continue
                pa, pb = a[f"{stat}_us"], b[f"{stat}_us"]
                delta = (pb - pa) / pa * 100.0
                f_new = floor_pct(new, proto, i, stat)
                f_prev = floor_pct(prev, proto, i, stat)
                floor = max(x for x in (f_new, f_prev, 0.0) if x is not None)
                resolved = abs(delta) > floor
                changed = i in CHANGED
                if resolved and changed:
                    v = "IMPROVED (expected)" if delta < 0 else "MOVED WRONG WAY"
                elif resolved and not changed:
                    v = "BLOCKER: control moved"
                    blockers.append((proto, stat, i + 1, delta, floor))
                elif not resolved and changed:
                    v = "under floor (no effect)"
                else:
                    v = "flat (expected)"
                rg = f"{RGROUP_PREV[i]}->{RGROUP_NOW[i]}" if changed else str(RGROUP_NOW[i])
                print(f"{i+1:>2} {labels[i]:>16} {rg:>5} {pa:>10.2f} "
                      f"{pb:>10.2f} {delta:>+8.2f} {floor:>7.2f} {v:>22}")
            ga = geo([cell(prev, proto, "ours", i)[f"{stat}_us"]
                      for i in range(6) if cell(prev, proto, "ours", i)])
            gb = geo([cell(new, proto, "ours", i)[f"{stat}_us"]
                      for i in range(6) if cell(new, proto, "ours", i)])
            if ga and gb:
                print(f"   geomean  prev {ga:.2f} us  ->  new {gb:.2f} us  "
                      f"({(gb - ga) / ga * 100:+.2f}%)")
                verdicts[(proto, stat)] = (ga, gb)

    print("\n################ null floors: cyclic order -> shuffled order ################")
    print("The rotation change's own detector. exp_26 saw a defective cyclic")
    print("order inflate its null to -2.29%; a shuffle collapsed it to -0.61%.")
    print(f"{'#':>2} {'shape':>16} {'proto':>10} {'prev%':>7} {'new%':>7} {'move':>9}")
    for proto in ("graded", "pipelined"):
        for i in range(6):
            fp, fn = floor_pct(prev, proto, i, "best"), floor_pct(new, proto, i, "best")
            if fp is None or fn is None:
                continue
            print(f"{i+1:>2} {labels[i]:>16} {proto:>10} {fp:>7.2f} {fn:>7.2f} "
                  f"{fn - fp:>+9.2f}")
    for proto in ("graded", "pipelined"):
        fp = [floor_pct(prev, proto, i, "best") for i in range(6)]
        fn = [floor_pct(new, proto, i, "best") for i in range(6)]
        fp = [x for x in fp if x is not None]
        fn = [x for x in fn if x is not None]
        if fp and fn:
            print(f"  {proto}: mean floor {sum(fp)/len(fp):.2f}% -> "
                  f"{sum(fn)/len(fn):.2f}%")

    # Is a control move common-mode across ARMS? A shift that hits ours,
    # ours_null, reference, rank1 and the no-op floor alike is not a property of
    # our kernel -- it is the run's own per-call constant moving, and absolute us
    # is then not comparable across runs while within-run ratios still are. The
    # graded floor arm's rsd is 150-270% on the small shapes, so this is not a
    # hypothetical.
    print("\n################ is the move common-mode across arms? ################")
    print("If every arm including the no-op floor moved together, the shift is the")
    print("run's per-call constant, not the kernel. Ratios stay valid; us do not.")
    for proto in ("graded", "pipelined"):
        print(f"\n--- {proto} / best: delta% vs previous run, per arm ---")
        arms = ["ours", "ours_null", "reference", "rank1", "harness_floor"]
        print(f"{'#':>2} {'shape':>16} " + "".join(f"{a:>15}" for a in arms)
              + f"{'spread':>9}")
        for i in range(6):
            cells, row = [], f"{i+1:>2} {labels[i]:>16} "
            for arm in arms:
                a, b = cell(prev, proto, arm, i), cell(new, proto, arm, i)
                if not a or not b:
                    row += f"{'-':>15}"
                    continue
                d = (b["best_us"] - a["best_us"]) / a["best_us"] * 100.0
                cells.append(d)
                row += f"{d:>+14.2f}%"
            if cells:
                row += f"{max(cells) - min(cells):>8.2f}%"
            print(row)
        print("  a small spread with a large common value == the constant moved,")
        print("  not the kernel.")

    print("\n################ within-run ratios: prev vs new, per shape ################")
    print("The shift-immune statistic. Both numerator and denominator come from")
    print("the same pool, same run, same interleave.")
    for other in ("rank1", "reference"):
        for proto in ("graded", "pipelined"):
            key = f"ours_vs_{other}__{proto}__best"
            rn, rp = new["ratios"].get(key), prev["ratios"].get(key)
            if not rn or not rp:
                continue
            print(f"\n--- ours/{other}, {proto}, best ---")
            print(f"{'#':>2} {'shape':>16} {'prev':>9} {'new':>9} {'move%':>8}")
            for i in range(6):
                a = (rp.get("per_shape") or [None] * 6)[i]
                b = (rn.get("per_shape") or [None] * 6)[i]
                if not a or not b:
                    continue
                mv = (b["ratio"] - a["ratio"]) / a["ratio"] * 100.0
                print(f"{i+1:>2} {labels[i]:>16} {a['ratio']:>9.4f} "
                      f"{b['ratio']:>9.4f} {mv:>+8.2f}")
            print(f"   geomean {rp['geomean']:.4f}x -> {rn['geomean']:.4f}x  "
                  f"({(rn['geomean'] - rp['geomean']) / rp['geomean'] * 100:+.2f}%)")

    print("\n################ the ratios that matter ################")
    for other in ("rank1", "reference"):
        for proto in ("graded", "pipelined"):
            for stat in ("best", "median"):
                key = f"ours_vs_{other}__{proto}__{stat}"
                rn = new["ratios"].get(key) or {}
                rp = prev["ratios"].get(key) or {}
                if not rn.get("geomean") or not rp.get("geomean"):
                    continue
                gn, gp = rn["geomean"], rp["geomean"]
                move = (gn - gp) / gp * 100.0
                call = ("unchanged (inside the 2% ratio floor)"
                        if abs(move) < RATIO_NOISE_FLOOR_PCT
                        else ("GAP NARROWED" if gn < gp else "GAP WIDENED"))
                print(f"  ours/{other:<9} {proto:>9}/{stat:<6}  "
                      f"prev {gp:.4f}x -> new {gn:.4f}x  ({move:+.2f}%)  {call}")
                if other == "rank1" and proto == "graded" and stat == "best":
                    proj = EXP26_PROJECTED_GRADED_GEOMEAN_RATIO
                    err = (gn - proj) / proj * 100.0
                    held = abs(err) < RATIO_NOISE_FLOOR_PCT
                    print(f"      exp_26 projected {proj:.4f}x from a PIPELINED "
                          f"measurement; measured graded {gn:.4f}x "
                          f"({err:+.2f}% off) -> projection "
                          f"{'HELD' if held else 'did NOT hold'}")
                    for i in (4,):
                        c = (rn.get("per_shape") or [None] * 6)[i]
                        if c:
                            print(f"      shape 5 graded ratio now {c['ratio']:.4f}x "
                                  f"(exp_26 projected ~1.20x)")

    print("\n################ pre-registration scorecard ################")
    if blockers:
        print(f"  BLOCKERS: {len(blockers)} control shape(s) moved outside floor")
        for proto, stat, sh, d, f in blockers:
            print(f"    shape {sh} {proto}/{stat}: {d:+.2f}% vs floor {f:.2f}%")
        print("  NOTE: this run changed TWO things (module + arm ordering), so a")
        print("  control move is attributable to either. LAD_ROT=cyclic on this")
        print("  same binary separates them in one run.")
    else:
        print("  no control shape moved outside its floor -- pre-registration held")
    json.dump({"blockers": blockers,
               "geomeans": {f"{p}_{s}": {"prev": v[0], "new": v[1]}
                            for (p, s), v in verdicts.items()}},
              open(os.path.join(D, "remeasure_delta.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
