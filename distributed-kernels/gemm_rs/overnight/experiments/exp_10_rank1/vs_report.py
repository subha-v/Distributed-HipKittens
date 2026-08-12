"""Aggregate the interleaved ours-vs-rank1 per-rank JSON into one table.

Reports best, median and mean per shape per arm plus the geometric means, and a
per-rank spread column, because relative stdevs on this node reach 90%+ on some
arms and a single mean is not a result.
"""

import glob
import json
import math
import os
import statistics
import sys

ARMS = ["ours", "rank1"]
LABEL = {"ours": "ours", "rank1": "rank-1"}


def geo(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    root = sys.argv[1]
    shapes = {}
    for path in sorted(glob.glob(os.path.join(root, "vs_s*.rank*.json"))):
        base = os.path.basename(path)
        index = int(base.split(".")[0].replace("vs_s", ""))
        rank = int(base.split(".")[1].replace("rank", ""))
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not data:
            continue
        shapes.setdefault(index, {})[rank] = data

    if not shapes:
        print("no results found in", root)
        return 1

    print()
    print("Per-shape, pooled over all 8 ranks (the graded number is a "
          "per-call wall time, so every rank's samples count).")
    print()
    header = (f"{'#':>2} {'shape':>21} {'arm':>7} {'best':>9} {'median':>9} "
              f"{'mean':>9} {'sd%':>6} {'worst':>10} {'n':>5}  ratio(ours/r1)")
    print(header)
    print("-" * len(header))

    agg = {arm: {"best": [], "median": [], "mean": []} for arm in ARMS}
    correctness_rows = []
    for index in sorted(shapes):
        per_rank = shapes[index]
        any_rank = next(iter(per_rank.values()))
        m, n, k, has_bias = any_rank["shape"]
        stats = {}
        for arm in ARMS:
            pooled = []
            for rank in sorted(per_rank):
                pooled.extend(per_rank[rank]["arms"].get(arm, []))
            if not pooled:
                continue
            stats[arm] = {
                "best": min(pooled),
                "median": statistics.median(pooled),
                "mean": statistics.mean(pooled),
                "sd": (statistics.stdev(pooled) / statistics.mean(pooled) * 100
                       if len(pooled) > 1 else 0.0),
                "worst": max(pooled),
                "n": len(pooled),
            }
        for arm in ARMS:
            if arm not in stats:
                continue
            s = stats[arm]
            ratio = ""
            if arm == "rank1" and "ours" in stats:
                r_best = stats["ours"]["best"] / s["best"]
                r_mean = stats["ours"]["mean"] / s["mean"]
                ratio = f"  best {r_best:.3f}x  mean {r_mean:.3f}x"
            print(f"{index+1:>2} {f'{m}x{n}x{k}':>21} {LABEL[arm]:>7} "
                  f"{s['best']:9.2f} {s['median']:9.2f} {s['mean']:9.2f} "
                  f"{s['sd']:6.1f} {s['worst']:10.2f} {s['n']:>5}{ratio}")
            for key in ("best", "median", "mean"):
                agg[arm][key].append(s[key])
        # correctness, worst over ranks
        for arm in ARMS:
            worst = None
            ok1 = ok2 = True
            for rank in sorted(per_rank):
                c = per_rank[rank].get("correctness", {}).get(arm)
                if not c:
                    continue
                ok1 &= c["allclose_1e-2"]
                ok2 &= c["allclose_2e-3"]
                worst = max(worst or 0.0, c["max_abs_diff"])
            correctness_rows.append(
                (index + 1, f"{m}x{n}x{k}", LABEL[arm], ok1, ok2, worst))
        print()

    print("=" * 78)
    print("GEOMETRIC MEANS (the competition's ranking statistic)")
    print()
    print(f"{'arm':>8} {'geo best':>11} {'geo median':>12} {'geo mean':>11}")
    for arm in ARMS:
        print(f"{LABEL[arm]:>8} {geo(agg[arm]['best']):11.2f} "
              f"{geo(agg[arm]['median']):12.2f} {geo(agg[arm]['mean']):11.2f}")
    if all(agg[arm]["best"] for arm in ARMS):
        for key in ("best", "median", "mean"):
            ours, r1 = geo(agg["ours"][key]), geo(agg["rank1"][key])
            verdict = "OURS FASTER" if ours < r1 else "rank-1 FASTER"
            print(f"  geo {key:>6}: ours/rank-1 = {ours / r1:.3f}x  "
                  f"-> {verdict}")
    print()
    print("=" * 78)
    print("CORRECTNESS (worst over ranks; both tolerances)")
    print(f"{'#':>2} {'shape':>21} {'arm':>7} {'1e-2':>6} {'2e-3':>6} "
          f"{'max|diff|':>11}")
    for row in correctness_rows:
        index, shape, arm, ok1, ok2, worst = row
        print(f"{index:>2} {shape:>21} {arm:>7} {str(ok1):>6} {str(ok2):>6} "
              f"{(worst if worst is not None else float('nan')):11.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
