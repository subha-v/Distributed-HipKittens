#!/usr/bin/env python3
"""exp_03 verdict from the rank-0 JSONs.

Usage: report_ab.py <logs_dir>

Per (shape, allocation order, protocol): within-round paired deltas
cmid-vs-base and null-vs-base, median delta %, wins/rounds, and an exact sign
test. Ship rule (plan.md): cmid must win in BOTH orders with the null contrast
inside its own spread.
"""
import glob
import json
import math
import statistics
import sys


def sign_test_p(wins, n):
    """Two-sided exact binomial sign test at p=0.5."""
    if n == 0:
        return 1.0
    def binom(k):
        return math.comb(n, k) / 2 ** n
    k = min(wins, n - wins)
    p = sum(binom(i) for i in range(0, k + 1)) * 2
    return min(1.0, p)


def main(logs):
    files = sorted(glob.glob(f"{logs}/ab_s*_*.rank0.json"))
    if not files:
        print("no rank0 JSONs found")
        return 1
    for path in files:
        d = json.load(open(path))
        if d.get("error"):
            print(f"{path}: ERROR\n{d['error'][:500]}")
            continue
        label = d["shape_label"]
        order = d["alloc_order"]
        pr = d["per_round"]
        print(f"\n=== {label}  alloc_order={order}  rounds={d['rounds']} "
              f"bit_identical={d['all_ranks_bit_identical']} ===")
        for proto in ("graded", "pipelined"):
            base = pr["base"][proto]
            for arm in ("cmid", "null"):
                a = pr[arm][proto]
                n = min(len(a), len(base))
                deltas = [(a[i] - base[i]) / base[i] * 100 for i in range(n)]
                wins = sum(1 for x in deltas if x < 0)
                med = statistics.median(deltas)
                p = sign_test_p(wins, n)
                print(f"  {proto:>9} {arm:>4} vs base: median {med:+.2f}%  "
                      f"wins {wins}/{n}  sign-p {p:.2e}  "
                      f"(base med {statistics.median(base):.1f} us, "
                      f"{arm} med {statistics.median(a):.1f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "logs"))
