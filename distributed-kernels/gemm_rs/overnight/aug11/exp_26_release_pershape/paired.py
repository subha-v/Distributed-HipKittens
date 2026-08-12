"""Per-ROUND paired analysis of the exp_26 A/B, and the null arm beside it.

Why this and not best-of-pass. The four passes disagree about which arm is
fastest on 8192x4096x14336, and the null arm -- two builds of the SAME rule --
moved -2.36%, -1.58%, +3.19%, -4.40% across them. That between-pass mode is
larger than anything being measured, and best-of-pass cannot see past it: it
compares one arm's luckiest round against another arm's luckiest round, in
different draws.

The A/B was built to avoid exactly this. Inside one round every arm's timed
block runs back to back, in a rotating order, over the same allocation and the
same clock state. So the right statistic is the WITHIN-ROUND difference, which
cancels the pass mode by construction, pooled over every round of every pass.
The null arm goes through the identical treatment and is the only thing that
says whether a within-round difference means anything.

  python3 paired.py <logs_dir>
"""

import glob
import json
import math
import os
import statistics
import sys

ARMS = ["ps0b", "ps1", "ps2", "rg2c"]
BASE = "ps0"
NAMES = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
         "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]
# What each arm's rgroup is on each shape, from the source rules and the
# tiles-per-CTA vector 1/1/1/1/2/4. Arms marked "=" are the same instruction
# stream as ps0 on that shape and are therefore controls, not candidates.
RG = {
    "ps0":  [1, 1, 1, 1, 1, 4],
    "ps0b": [1, 1, 1, 1, 1, 4],
    "ps1":  [1, 1, 1, 1, 2, 4],
    "ps2":  [1, 1, 1, 1, 2, 4],
    "rg2c": [1, 1, 1, 1, 2, 2],
}


def sign_test(deltas):
    """Two-sided sign test p-value for 'the median difference is zero'."""
    n = sum(1 for d in deltas if d != 0)
    k = sum(1 for d in deltas if d < 0)
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    logs = sys.argv[1] if len(sys.argv) > 1 else "logs"
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    runs = []
    for path in sorted(glob.glob(os.path.join(logs, "ab_pershape_*.json"))):
        tag = os.path.basename(path)[12:-5]
        if only and not tag.startswith(only):
            continue
        runs.append((tag, json.load(open(path))))
    if not runs:
        print(f"no A/B json under {logs}" + (f" matching {only}" if only else ""))
        return
    print(f"passes pooled: {', '.join(t for t, _ in runs)}")

    keys = list(runs[0][1]["per_shape"])
    print("\nWITHIN-ROUND PAIRED DIFFERENCE vs ps0, pooled over every round of "
          "every pass")
    print("  n      = paired rounds;  median/mean = of the per-round % deltas")
    print("  wins   = rounds where the arm was faster than ps0 in that same round")
    print("  p      = two-sided sign test;  '=' marks an arm with ps0's rgroup "
          "on that shape (a control)\n")

    for i, k in enumerate(keys):
        ppc = runs[0][1]["per_shape"][k]["arms"]["_geometry"]["tiles_per_cta"]
        print(f"--- shape {i+1}: {NAMES[i]}  ({ppc} tile(s) per CTA) ---")
        print(f"    {'arm':<6}{'rgroup':>7}{'n':>5}{'median%':>10}{'mean%':>9}"
              f"{'wins':>10}{'p':>9}")
        for arm in ARMS:
            deltas = []
            for _, r in runs:
                a = r["per_shape"][k]["arms"]
                if arm not in a or BASE not in a:
                    continue
                for x, y in zip(a[arm]["samples"], a[BASE]["samples"]):
                    deltas.append((x / y - 1.0) * 100.0)
            if not deltas:
                continue
            wins = sum(1 for d in deltas if d < 0)
            ctl = "=" if RG[arm][i] == RG[BASE][i] else " "
            print(f"    {arm:<6}{str(RG[arm][i]) + ctl:>7}{len(deltas):>5}"
                  f"{statistics.median(deltas):>+10.2f}"
                  f"{statistics.mean(deltas):>+9.2f}"
                  f"{wins}/{len(deltas):>8}"
                  f"{sign_test(deltas):>9.4f}")
        print()

    print("=" * 78)
    print("READ THIS FIRST: the null arm ps0b is byte-equivalent to ps0 in "
          "behaviour.\nAny arm whose paired median is not clearly outside "
          "ps0b's is not distinguishable\nfrom a second build of the incumbent, "
          "however tidy its point estimate looks.")


if __name__ == "__main__":
    main()
