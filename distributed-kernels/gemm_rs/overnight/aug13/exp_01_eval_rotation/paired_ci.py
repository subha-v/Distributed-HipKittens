#!/usr/bin/env python3
"""exp_01 order-balanced table with paired CIs.

Usage: paired_ci.py <raw_dir>

Consumes the raw/<session>/<arm>/benchmark.popcorn.txt tree (same parser
contract as parse_popcorn.py: refuse files without `check: pass` or with
fewer than 6 benchmark blocks). Produces:

  1. per-arm per-session GM(best) and the pooled per-shape best table;
  2. WITHIN-SESSION paired ratios O/K, O/R, O/D as geomeans of per-shape
     best, with a t-based CI over sessions on the log scale (the pairing
     is the session: both arms share clocks, boot state and drift);
  3. per-shape paired ratio CIs (same pairing);
  4. Q3 drift: per-arm max/min GM spread across sessions;
  5. order check: arm-first vs arm-not-first GM contrast per arm.

CI note: n is small (5 sessions; 2 for D). The t quantile is looked up from
a fixed table (df 1..9) -- no scipy on the node.
"""
import math
import re
import sys
from pathlib import Path

SHAPES = ["64", "512", "2048", "4096", "8192a", "8192b"]
# two-sided 95% t quantiles, df 1..9
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def parse_file(path):
    text = path.read_text()
    if "check: pass" not in text:
        return None, f"no 'check: pass' in {path}"
    fields = {}
    for line in text.splitlines():
        m = re.match(r"benchmark\.(\d+)\.(\w+): (.*)", line)
        if m:
            fields.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3)
    if len(fields) < 6:
        return None, f"only {len(fields)} benchmark blocks in {path}"
    return [float(fields[i]["best"]) / 1000.0 for i in range(6)], None


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def mean_sd(v):
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))


def ratio_ci(logs):
    """mean ratio and 95% CI from per-session log-ratios."""
    n = len(logs)
    m, sd = mean_sd(logs)
    if n < 2 or math.isnan(sd):
        return math.exp(m), None, None, n
    half = T95[n - 1] * sd / math.sqrt(n)
    return math.exp(m), math.exp(m - half), math.exp(m + half), n


def main(raw):
    raw = Path(raw)
    data = {}  # (session, arm) -> [6 best us]
    # the K arm's popcorn is `bench.popcorn.txt` (rank1_pass label), the
    # others are `benchmark.popcorn.txt` -- glob both.
    for pop in sorted(raw.glob("*/*/*.popcorn.txt")):
        session, arm = pop.parent.parent.name, pop.parent.name
        if arm not in ("O", "R", "K", "D"):
            continue
        best, err = parse_file(pop)
        if err:
            print(f"SKIP {err}")
            continue
        data[(session, arm)] = best

    sessions = sorted({s for s, _ in data})
    arms = sorted({a for _, a in data})

    print("== 1. per-session GM(best), us ==")
    print(f"{'sess':>5} " + " ".join(f"{a:>8}" for a in arms))
    for s in sessions:
        row = [f"{geomean(data[(s, a)]):8.1f}" if (s, a) in data else f"{'-':>8}"
               for a in arms]
        print(f"{s:>5} " + " ".join(row))

    print("\n== pooled per-shape best (min over sessions), us ==")
    print(f"{'arm':>4} " + " ".join(f"{sh:>8}" for sh in SHAPES) + f" {'GM':>8}")
    for a in arms:
        per = [[data[(s, a)][i] for s in sessions if (s, a) in data]
               for i in range(6)]
        if not per[0]:
            continue
        best = [min(v) for v in per]
        print(f"{a:>4} " + " ".join(f"{b:8.1f}" for b in best) +
              f" {geomean(best):8.1f}")

    print("\n== 2. within-session paired ratios (GM of per-shape best) ==")
    for num, den in (("O", "K"), ("O", "R"), ("O", "D")):
        logs = [math.log(geomean(data[(s, num)]) / geomean(data[(s, den)]))
                for s in sessions
                if (s, num) in data and (s, den) in data]
        if not logs:
            continue
        r, lo, hi, n = ratio_ci(logs)
        ci = f"  95% CI [{lo:.4f}, {hi:.4f}]" if lo else "  (n<2, no CI)"
        print(f"  {num}/{den} = {r:.4f}  n={n} sessions{ci}")

    print("\n== 3. per-shape paired ratio CIs ==")
    for num, den in (("O", "K"), ("O", "R")):
        print(f"  {num}/{den}:")
        for i, sh in enumerate(SHAPES):
            logs = [math.log(data[(s, num)][i] / data[(s, den)][i])
                    for s in sessions
                    if (s, num) in data and (s, den) in data]
            if not logs:
                continue
            r, lo, hi, n = ratio_ci(logs)
            ci = f" [{lo:.3f}, {hi:.3f}]" if lo else ""
            print(f"    {sh:>6}: {r:.4f}{ci}  n={n}")

    print("\n== 4. Q3 drift: per-arm GM spread across sessions ==")
    for a in arms:
        gms = [geomean(data[(s, a)]) for s in sessions if (s, a) in data]
        if len(gms) < 2:
            continue
        print(f"  {a}: min {min(gms):.1f}  max {max(gms):.1f}  "
              f"spread {(max(gms) / min(gms) - 1) * 100:.2f}%  n={len(gms)}")

    print("\n== 5. order check: GM when arm ran first vs later ==")
    first_arm = {}
    for s in sessions:
        arms_in = [(a, data[(s, a)]) for a in arms if (s, a) in data]
        # session order is not recoverable from the tree alone; the driver's
        # session() order is fixed per name -- encode it here.
        order = {"s0": "ORK", "s1": "KROD", "s2": "DORK",
                 "s3": "KOR", "s4": "OKR"}.get(s)
        if order:
            first_arm[s] = order[0]
    for a in arms:
        first = [geomean(data[(s, a)]) for s in sessions
                 if (s, a) in data and first_arm.get(s) == a]
        later = [geomean(data[(s, a)]) for s in sessions
                 if (s, a) in data and first_arm.get(s, a) != a]
        if first and later:
            f, l = geomean(first), geomean(later)
            print(f"  {a}: first {f:.1f} (n={len(first)})  "
                  f"later {l:.1f} (n={len(later)})  "
                  f"first/later {f / l:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw")
