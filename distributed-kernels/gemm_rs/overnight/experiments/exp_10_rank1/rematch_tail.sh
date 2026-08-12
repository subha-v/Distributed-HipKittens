#!/usr/bin/env bash
# Shape 3 (2048x2880x2880) came in with mean 327.77 vs median 218.46 and a worst
# of 1771.89 us on our arm, which is what drags the mean-based geomean from
# 1.27x to 1.36x. Is that a handful of stray calls or a genuine bimodality? If it
# is a few samples, the median is the statistic to quote and the mean is
# contaminated; if it is half the distribution, it is a real kernel behaviour.
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1

python3 - "$E/vs_logs" <<'PY'
import glob, json, os, statistics, sys
root = sys.argv[1]
per = {}
for p in sorted(glob.glob(os.path.join(root, "vs_s*.rank*.json"))):
    b = os.path.basename(p)
    idx = int(b.split(".")[0].replace("vs_s", ""))
    rk = int(b.split(".")[1].replace("rank", ""))
    d = json.load(open(p))
    if not d:
        continue
    per.setdefault(idx, {}).setdefault(rk, d)

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]

print(f"{'#':>2} {'arm':>7} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9} "
      f"{'>2x med':>8} {'>1.3x med':>10}")
for idx in sorted(per):
    for arm in ("ours", "rank1"):
        xs = []
        for rk in sorted(per[idx]):
            xs += per[idx][rk]["arms"].get(arm, [])
        if not xs:
            continue
        med = statistics.median(xs)
        n2 = sum(1 for x in xs if x > 2 * med)
        n13 = sum(1 for x in xs if x > 1.3 * med)
        print(f"{idx+1:>2} {arm:>7} {med:9.2f} {pct(xs,0.90):9.2f} "
              f"{pct(xs,0.99):9.2f} {max(xs):9.2f} {n2:>8} {n13:>10}")
    print()

print("--- shape 3 (index 2), our arm, where do the outliers sit? ---")
d3 = per.get(2, {})
for rk in sorted(d3):
    xs = d3[rk]["arms"]["ours"]
    med = statistics.median(xs)
    bad = [(i, round(x, 1)) for i, x in enumerate(xs) if x > 2 * med]
    print(f"  rank{rk}: n={len(xs)} med={med:7.2f} outliers(>2x med, "
          f"index within rank)={bad}")
print()
print("sample index 0-11 is rep0, 12-23 is rep1; the arm order reverses between")
print("reps, so an outlier at index 0 or 12 is a first-call-after-arm-switch")
print("effect rather than steady-state kernel behaviour.")

print()
print("--- geomean recomputed on medians and on a trimmed mean (drop top 5%) ---")
import math
def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))
for label, fn in (("median", lambda xs: statistics.median(xs)),
                  ("mean", lambda xs: statistics.mean(xs)),
                  ("trimmed mean p95", lambda xs: statistics.mean(
                      sorted(xs)[:max(1, int(0.95 * len(xs)))]))):
    vals = {}
    for arm in ("ours", "rank1"):
        acc = []
        for idx in sorted(per):
            xs = []
            for rk in sorted(per[idx]):
                xs += per[idx][rk]["arms"].get(arm, [])
            acc.append(fn(xs))
        vals[arm] = geo(acc)
    print(f"  {label:>17}: ours={vals['ours']:8.2f} rank1={vals['rank1']:8.2f} "
          f"ratio={vals['ours']/vals['rank1']:.3f}x")
PY
echo "===== DONE ====="
