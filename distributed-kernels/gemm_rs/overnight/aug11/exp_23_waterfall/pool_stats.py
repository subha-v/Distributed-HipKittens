"""exp_23 scorer: paired within-draw contrasts against a measured null set.

The unit of evidence is ONE DRAW, not one sample. Every sample inside a process
shares that process's hipMalloc outcome, so samples within a draw are repeated
measurements of one allocation, not independent observations of the arm. A test
that treats them as n independent samples will happily declare significance on
a 4% allocation artefact. (`experiments/exp_14_tile_waves/pool.py` argues this
at length; its exact rank-sum scorer is reused here verbatim in spirit.)

So each draw contributes ONE number per contrast:

    rung b   g = median(a)    / median(b)    - 1     (positive = b faster)
    rung c   g = median(b)    / median(c)    - 1
    NR=X     g = median(c)    / median(nrX)  - 1
    NULL     g* = median(c)   / median(null) - 1     <- the measured null set

`null` is the same device ISA as `c` (fingerprint assertion A4) with an
independent allocation, so g* is a direct sample of "difference of two
allocation biases plus within-draw jitter" -- exactly the null distribution the
other contrasts are drawn from when their mechanism is inert. Under that null
the two sets are exchangeable, so D candidate contrasts all beyond D null
contrasts is one of C(2D, D) arrangements.

RESOLUTION LIMIT, stated before the numbers rather than after: at D draws per
side the smallest attainable one-sided p is 1/C(2D, D). At D = 4 that is
1/70 = 0.0143. A contrast that does not separate at four draws is reported
UNRESOLVED and left there; adding draws until it crosses is how a false
positive gets manufactured.

usage: pool_stats.py [draw_*.json ...]
"""

import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, baseline_label) for every contrast that gets a verdict.
RUNG_CONTRASTS = [("b", "a"), ("c", "b")]
NR_POINTS = [8, 16, 32, 48]
NULL_CONTRAST = ("null", "c")
# Contrasts the mechanism predicts are inert on some shapes; see plan.md. These
# are the structural check, and a resolvable delta in one of them is a blocker.
PREDICTED_FLAT_KEY = {"b": "ab_rung_active", "c": "bc_rung_active"}


def rank_sum_p(a, b):
    """Exact one-sided Wilcoxon rank-sum p for "a is larger than b", no ties.

    Full-range disjointness gets STRICTER as draws accumulate, because a range
    only grows, so a real effect can be "confirmed" at 4 draws and
    "unconfirmed" at 8 on the same data. The rank-sum uses all the ordering
    information rather than only the two extremes and is monotone in evidence.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    values = sorted(a + b)
    ranks = {v: i + 1 for i, v in enumerate(values)}
    observed = sum(ranks[v] for v in a)
    total_n = n1 + n2
    dp = [[0] * (total_n * (total_n + 1) // 2 + 1) for _ in range(n1 + 1)]
    dp[0][0] = 1
    for r in range(1, total_n + 1):
        for k in range(min(n1, r), 0, -1):
            for s in range(len(dp[k]) - 1, r - 1, -1):
                if dp[k - 1][s - r]:
                    dp[k][s] += dp[k - 1][s - r]
    atleast = sum(dp[n1][s] for s in range(observed, len(dp[n1])))
    return atleast / math.comb(total_n, n1)


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def contrast(res, label, base, field):
    """Within-draw gain of `label` over `base`; positive means label is faster."""
    arm, ref = res["arms"].get(label), res["arms"].get(base)
    if not arm or not ref or arm[field] is None or ref[field] is None:
        return None
    return (ref[field] / arm[field] - 1) * 100.0


def score(sets, nulls, n_draws):
    """Disjointness + exact rank-sum on one contrast against the null set."""
    gb, gm = sets["best"], sets["median"]
    nb, nm = nulls["best"], nulls["median"]
    out = {
        "draws": len(gm),
        "gain_best_pct": statistics.median(gb) if gb else None,
        "gain_median_pct": statistics.median(gm) if gm else None,
        "gain_best_range": [min(gb), max(gb)] if gb else None,
        "gain_median_range": [min(gm), max(gm)] if gm else None,
        "null_best_range": [min(nb), max(nb)] if nb else None,
        "null_median_range": [min(nm), max(nm)] if nm else None,
    }
    if not gm or not nm:
        out["verdict"] = "no data"
        return out
    n = min(len(gm), len(nm))
    faster = (min(gb[:n]) > max(nb[:n])) and (min(gm[:n]) > max(nm[:n]))
    slower = (max(gb[:n]) < min(nb[:n])) and (max(gm[:n]) < min(nm[:n]))
    out["p_ranksum_faster"] = rank_sum_p(gm[:n], nm[:n])
    out["p_ranksum_slower"] = rank_sum_p(nm[:n], gm[:n])
    out["p_floor_at_this_draw_count"] = 1.0 / math.comb(2 * n, n)
    if faster:
        out["verdict"] = "RESOLVED faster"
    elif slower:
        out["verdict"] = "RESOLVED slower"
    else:
        out["verdict"] = "UNRESOLVED"
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "draw_*.json")))
    if not paths:
        print("no draw_*.json found")
        return 1
    draws = [(os.path.basename(p), json.load(open(p))) for p in sorted(paths)]
    n_rev = sum(1 for _, d in draws if d.get("reverse"))
    print(f"pooling {len(draws)} allocation draw(s): "
          f"{len(draws) - n_rev} forward construction, {n_rev} reversed")
    for name, _ in draws:
        print(f"  {name}")
    print(f"\nresolution limit at {len(draws)} draws per side: smallest "
          f"attainable one-sided p = 1/C({2 * len(draws)},{len(draws)}) = "
          f"{1.0 / math.comb(2 * len(draws), len(draws)):.4f}")

    indices = sorted({int(k) for _, d in draws for k in d["shapes"]})
    report = {"draws": [n for n, _ in draws], "draws_reversed": n_rev,
              "shapes": {}, "blockers": []}

    for index in indices:
        key = str(index)
        present = [(n, d["shapes"][key]) for n, d in draws if key in d["shapes"]]
        if not present:
            continue
        tag = present[0][1]["shape"]
        meta = present[0][1]

        # The instrument's own resolution at this shape, measured TONIGHT.
        floors = []
        for _, res in present:
            c, nul = res["arms"].get("c"), res["arms"].get("null")
            if c and nul and c["median_us"] and nul["median_us"]:
                floors.append(abs(c["median_us"] - nul["median_us"]) /
                              min(c["median_us"], nul["median_us"]) * 100)
        floor = max(floors) if floors else float("nan")

        nulls = {f: [contrast(res, *NULL_CONTRAST, f"{f}_us")
                     for _, res in present] for f in ("best", "median")}
        nulls = {f: [v for v in vals if v is not None]
                 for f, vals in nulls.items()}

        shape_out = {
            "shape": tag,
            "null_floor_worst_draw_pct": floor,
            "null_floor_per_draw_pct": floors,
            "ab_rung_active": meta.get("ab_rung_active"),
            "bc_rung_active": meta.get("bc_rung_active"),
            "shipped_nr": meta.get("shipped_nr"),
            "arms": {}, "contrasts": {},
        }

        labels = list(present[0][1]["arms"].keys())
        for label in labels:
            bests = [res["arms"][label]["best_us"] for _, res in present
                     if res["arms"].get(label, {}).get("best_us")]
            meds = [res["arms"][label]["median_us"] for _, res in present
                    if res["arms"].get(label, {}).get("median_us")]
            dead = [res["arms"][label].get("dead") for _, res in present
                    if res["arms"].get(label, {}).get("dead")]
            if not meds:
                shape_out["arms"][label] = {"dead": dead[:1]}
                continue
            shape_out["arms"][label] = {
                "best_us": min(bests), "median_us": statistics.median(meds),
                "per_draw_best_us": bests, "per_draw_median_us": meds,
                "dead": dead or None,
            }

        print(f"\n{'=' * 108}")
        print(f"shape {index + 1}: {tag}   draws={len(present)}   "
              f"NULL FLOOR (worst draw, c vs null) = {floor:.2f}%   "
              f"per draw: {' '.join(f'{v:.2f}' for v in floors)}")
        print(f"  a->b mechanism active here: {meta.get('ab_rung_active')}   "
              f"b->c mechanism active here: {meta.get('bc_rung_active')}   "
              f"shipped NR = {meta.get('shipped_nr')}")
        print(f"{'=' * 108}")
        head = (f"  {'contrast':>14}{'gain_b%':>9}{'gain_m%':>9}"
                f"{'m_lo':>8}{'m_hi':>8}{'null_lo':>9}{'null_hi':>9}"
                f"{'p_fast':>8}{'p_slow':>8}  verdict")
        print(head)

        def emit(name, label, base, predicted_flat=None):
            sets = {f: [v for v in (contrast(res, label, base, f"{f}_us")
                                    for _, res in present) if v is not None]
                    for f in ("best", "median")}
            s = score(sets, nulls, len(present))
            s["predicted_flat"] = predicted_flat
            s["beyond_floor"] = (abs(s["gain_median_pct"]) > floor
                                 if s.get("gain_median_pct") is not None
                                 else None)
            shape_out["contrasts"][name] = s
            if s.get("gain_median_pct") is None:
                print(f"  {name:>14}   no data")
                return
            mlo, mhi = s["gain_median_range"]
            nlo, nhi = s["null_median_range"]
            flag = ""
            if predicted_flat and s["verdict"].startswith("RESOLVED"):
                flag = "  <-- BLOCKER: predicted inert, resolved anyway"
                report["blockers"].append(
                    f"shape {index + 1} {tag}: contrast {name} predicted inert "
                    f"({s['gain_median_pct']:+.2f}% median) but {s['verdict']}")
            print(f"  {name:>14}{s['gain_best_pct']:>9.2f}"
                  f"{s['gain_median_pct']:>9.2f}{mlo:>8.2f}{mhi:>8.2f}"
                  f"{nlo:>9.2f}{nhi:>9.2f}"
                  f"{s['p_ranksum_faster']:>8.4f}{s['p_ranksum_slower']:>8.4f}"
                  f"  {s['verdict']}{flag}")

        for label, base in RUNG_CONTRASTS:
            active = meta.get(PREDICTED_FLAT_KEY[label])
            emit(f"{label} vs {base}", label, base,
                 predicted_flat=(active is False))
        emit("null vs c", "null", "c")
        for nr in NR_POINTS:
            emit(f"nr{nr} vs c", f"nr{nr}", "c")

        report["shapes"][str(index + 1)] = shape_out

    # Geomeans over the shapes present, from the pooled per-shape numbers.
    print(f"\n{'=' * 108}")
    print("GEOMEANS over the shapes present (best = min over all draws, "
          "median = median of per-draw medians)")
    print(f"{'=' * 108}")
    print(f"  {'arm':>8}{'geo_best_us':>14}{'geo_median_us':>15}"
          f"{'vs_a_best':>11}{'vs_a_med':>11}")
    geo = {}
    all_labels = list(next(iter(report["shapes"].values()))["arms"].keys())
    for label in all_labels:
        bests, meds = [], []
        for sh in report["shapes"].values():
            arm = sh["arms"].get(label, {})
            if arm.get("median_us"):
                bests.append(arm["best_us"])
                meds.append(arm["median_us"])
        if len(meds) != len(report["shapes"]):
            continue
        geo[label] = {"geomean_best_us": geomean(bests),
                      "geomean_median_us": geomean(meds)}
    base = geo.get("a")
    for label, g in geo.items():
        rb = base["geomean_best_us"] / g["geomean_best_us"] if base else float("nan")
        rm = base["geomean_median_us"] / g["geomean_median_us"] if base else float("nan")
        print(f"  {label:>8}{g['geomean_best_us']:>14.2f}"
              f"{g['geomean_median_us']:>15.2f}{rb:>11.4f}{rm:>11.4f}")
    report["geomeans"] = geo

    if report["blockers"]:
        print(f"\n{'!' * 108}")
        print("STRUCTURAL PREDICTION VIOLATED -- the figure is suspect and no "
              "rung verdict may be published:")
        for line in report["blockers"]:
            print("  " + line)
        print(f"{'!' * 108}")
    else:
        print("\nstructural prediction HOLDS: every contrast the mechanism "
              "predicts inert is unresolved against tonight's null floor.")

    path = os.path.join(HERE, "stats.json")
    json.dump(report, open(path, "w"), indent=2, default=str)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
