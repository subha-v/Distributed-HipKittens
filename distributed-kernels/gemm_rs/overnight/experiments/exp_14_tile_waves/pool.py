"""Pool independent allocation draws and decide each candidate by DISJOINT RANGE.

Why a t-test would be wrong here. The dominant error term on this node is a
per-allocation offset: every sample inside one process shares one hipMalloc
draw, so samples within a draw are not independent observations of the arm --
they are repeated measurements of one draw. Averaging them tightens nothing that
matters, and any test treating them as n independent samples will declare
significance on a 4% allocation artefact.

So the unit of evidence is ONE DRAW = one number per arm. Two tests are reported
on that unit, and they answer different questions.

TEST 1, PAIRED (primary). The statistic is the within-draw contrast
`g = T/candidate - 1`, and its null distribution is measured, not assumed: the
identically configured twin gives `g* = T/T* - 1` in the very same draws. Under
the null both are "difference of two independent allocation biases plus
within-draw jitter", so the two sets are exchangeable, and D candidate contrasts
all beyond D twin contrasts is one of C(2D, D) arrangements -- p = 1/C(2D, D),
which is 1/252 at D=5 and 1/48620 at D=9. This is the right unit because a draw
also carries node-level state (clocks, thermals, whatever else moves between
processes) that shifts T and the candidate TOGETHER; the contrast cancels it,
the raw microsecond value does not.

The paired test is conservative in the one way that matters: g and g* share the
same b_T within a draw, so they move together, which makes disjointness harder
to reach rather than easier.

TEST 2, RAW RANGE (secondary, strictly more conservative). Same criterion on the
raw per-draw microseconds. This one charges the candidate for draw-to-draw
variation that T pays too, so it can report "overlap" for a real effect whose
size is below the node's between-process drift. Reported because it is the
weaker claim and it is honest to show when only the paired test carries a
verdict.

A candidate lands only on a FASTER paired verdict. Where the two tests disagree,
both are reported and the disagreement is the finding.

usage: pool.py [json ...]      (default: every sweep_*.json beside this file)
"""

import glob
import json
import math
import os
import statistics
import sys


def comb(n, k):
    return math.comb(n, k)


def rank_sum_p(a, b):
    """Exact one-sided Wilcoxon rank-sum p for "a is larger than b", no ties.

    Full-range disjointness (the other test here) has a defect worth naming: it
    gets STRICTER as draws accumulate, because a range only grows. So a real
    effect can be "confirmed" at 5 draws and "unconfirmed" at 10 on the same
    data. The rank-sum test uses all the ordering information instead of only
    the two extremes and is monotone in evidence, which is what a test should be.

    Computed exactly by counting, over all C(n1+n2, n1) label assignments, how
    many give a rank sum for `a` at least as extreme as observed.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    values = sorted(a + b)
    ranks = {v: i + 1 for i, v in enumerate(values)}
    observed = sum(ranks[v] for v in a)
    # counts[s] = number of ways to choose k of the n ranks summing to s
    total_n = n1 + n2
    # dp[k][s] over ranks 1..total_n
    dp = [[0] * (total_n * (total_n + 1) // 2 + 1) for _ in range(n1 + 1)]
    dp[0][0] = 1
    for r in range(1, total_n + 1):
        for k in range(min(n1, r), 0, -1):
            for s in range(len(dp[k]) - 1, r - 1, -1):
                if dp[k - 1][s - r]:
                    dp[k][s] += dp[k - 1][s - r]
    atleast = sum(dp[n1][s] for s in range(observed, len(dp[n1])))
    return atleast / comb(total_n, n1)


def load(paths):
    draws = []
    for path in sorted(paths):
        with open(path) as handle:
            draws.append((os.path.basename(path), json.load(handle)))
    return draws


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here,
                                                          "sweep_*.json")))
    if not paths:
        print("no sweep_*.json found")
        return 1
    draws = load(paths)
    print(f"pooling {len(draws)} independent allocation draws:")
    for name, _ in draws:
        print(f"  {name}")

    shape_indices = sorted({int(k) for _, d in draws for k in d["shapes"]})
    verdicts = {}
    for index in shape_indices:
        key = str(index)
        present = [(name, d["shapes"][key]) for name, d in draws
                   if key in d["shapes"]]
        if not present:
            continue
        tag = present[0][1]["shape"]
        labels, order = {}, []
        for _, res in present:
            for label in res["arms"]:
                if label not in labels:
                    labels[label] = True
                    order.append(label)

        print(f"\n{'=' * 104}")
        print(f"shape {index + 1}: {tag}   draws={len(present)}")
        print(f"{'=' * 104}")

        series = {}
        for label in order:
            best, med, dead, gain_b, gain_m = [], [], [], [], []
            for name, res in present:
                arm = res["arms"].get(label)
                if arm is None:
                    continue
                if arm["best_us"] is None:
                    dead.append(arm["dead"])
                    continue
                best.append(arm["best_us"])
                med.append(arm["median_us"])
                # The paired contrast, taken against T from the SAME process.
                ref = res["arms"]["T"]
                gain_b.append((ref["best_us"] / arm["best_us"] - 1) * 100)
                gain_m.append((ref["median_us"] / arm["median_us"] - 1) * 100)
            series[label] = {"best": best, "median": med, "dead": dead,
                             "gain_b": gain_b, "gain_m": gain_m}

        # The instrument's own floor, per draw: T vs its identically configured
        # twin. Reported as the WORST draw, because that is the number a delta
        # has to clear to be believable at all.
        floors_b, floors_m = [], []
        tb, tm = series["T"]["best"], series["T"]["median"]
        wb, wm = series.get("T*", {}).get("best", []), \
            series.get("T*", {}).get("median", [])
        for i in range(min(len(tb), len(wb))):
            floors_b.append(abs(tb[i] - wb[i]) / min(tb[i], wb[i]) * 100)
            floors_m.append(abs(tm[i] - wm[i]) / min(tm[i], wm[i]) * 100)
        floor_b = max(floors_b) if floors_b else float("nan")
        floor_m = max(floors_m) if floors_m else float("nan")
        # Pooled across draws, T and T* are 2D samples of the SAME
        # configuration, so their total spread is the honest floor for a
        # cross-draw comparison.
        pooled = tb + wb
        pooled_floor = ((max(pooled) - min(pooled)) / min(pooled) * 100
                        if pooled else float("nan"))
        print(f"  null floor, worst single draw: best {floor_b:.2f}% "
              f"median {floor_m:.2f}%   per-draw: "
              f"{' '.join(f'{v:.2f}' for v in floors_b)}")
        print(f"  null floor, POOLED T+T* range over all draws: "
              f"{pooled_floor:.2f}%  "
              f"[{min(pooled):.2f}, {max(pooled):.2f}] us")

        null_gains = series.get("T*", {}).get("gain_b", [])
        null_gains_m = series.get("T*", {}).get("gain_m", [])
        head = (f"  {'arm':>12}{'n':>3}{'best_min':>10}{'best_max':>10}"
                f"{'gain_b%':>9}{'gain_m%':>9}"
                f"{'gain_lo':>9}{'gain_hi':>9}{'PAIRED':>9}{'p_pair':>9}"
                f"{'p_rank':>9}{'rawrange':>10}")
        print("  " + "-" * (len(head) - 2))
        rev = sum(1 for _, res in present if res.get("reverse"))
        print(f"  construction order: {len(present) - rev} forward, "
              f"{rev} reversed")
        print(f"  null contrast set (T vs T*), best: "
              f"[{min(null_gains):+.2f}, {max(null_gains):+.2f}] % "
              f"median {statistics.median(null_gains):+.2f}%")
        print(head)
        for label in order:
            s = series[label]
            if not s["best"]:
                print(f"  {label:>12}{0:>3}   DEAD: {s['dead'][0]}")
                continue
            gb = statistics.median(s["gain_b"])
            gm = statistics.median(s["gain_m"])
            n = min(len(s["best"]), len(tb))
            # Test 2: raw per-draw microseconds.
            raw = ("FASTER" if max(s["best"][:n]) < min(tb[:n]) else
                   "SLOWER" if min(s["best"][:n]) > max(tb[:n]) else "overlap")
            # Test 1: paired contrast against the twin's contrast set.
            paired, p_pair, p_rank = "overlap", float("nan"), float("nan")
            if label not in ("T", "T*") and null_gains:
                nn = min(len(s["gain_b"]), len(null_gains))
                fast = (min(s["gain_b"][:nn]) > max(null_gains[:nn]) and
                        min(s["gain_m"][:nn]) > max(null_gains_m[:nn]))
                slow = (max(s["gain_b"][:nn]) < min(null_gains[:nn]) and
                        max(s["gain_m"][:nn]) < min(null_gains_m[:nn]))
                if fast or slow:
                    paired = "FASTER" if fast else "SLOWER"
                    p_pair = 1.0 / comb(2 * nn, nn)
                p_rank = rank_sum_p(s["gain_b"][:nn], null_gains[:nn])
            print(f"  {label:>12}{len(s['best']):>3}{min(s['best']):>10.2f}"
                  f"{max(s['best']):>10.2f}{gb:>9.2f}{gm:>9.2f}"
                  f"{min(s['gain_b']):>9.2f}{max(s['gain_b']):>9.2f}"
                  f"{paired:>9}{p_pair:>9.5f}{p_rank:>9.5f}{raw:>10}")
            if label not in ("T", "T*"):
                verdicts[(index, label)] = {
                    "gain_best_pct": gb, "gain_median_pct": gm,
                    "gain_best_range": [min(s["gain_b"]), max(s["gain_b"])],
                    "paired": paired, "p_paired": p_pair, "p_ranksum": p_rank,
                    "raw_range": raw, "draws": len(s["best"]),
                    "floor_worst_pct": floor_b,
                    "pooled_floor_pct": pooled_floor,
                    "null_contrast_range": [min(null_gains), max(null_gains)],
                    "null_contrast_median": statistics.median(null_gains),
                }

    print(f"\n{'=' * 104}")
    print("LANDING TABLE (FASTER disjoint paired range, or p_ranksum < 0.01 "
          "with a consistently positive sign)")
    print(f"{'=' * 104}")
    for (index, label), v in sorted(verdicts.items()):
        lo, hi = v["gain_best_range"]
        ok = (v["paired"] == "FASTER" or
              (v["p_ranksum"] < 0.01 and v["gain_best_pct"] > 0))
        print(f"  shape {index + 1} {label:<12} best {v['gain_best_pct']:+6.2f}% "
              f"median {v['gain_median_pct']:+6.2f}%  per-draw [{lo:+.2f},"
              f"{hi:+.2f}]  paired {v['paired']:<8} p_pair={v['p_paired']:.5f} "
              f"p_rank={v['p_ranksum']:.5f}  raw {v['raw_range']:<8} "
              f"{'LAND' if ok else 'do not land'}")
    out = os.path.join(here, "pooled.json")
    with open(out, "w") as handle:
        json.dump({f"{k[0] + 1}:{k[1]}": v for k, v in verdicts.items()},
                  handle, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
