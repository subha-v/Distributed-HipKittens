"""Is our shape-4 latency bimodal? Analyse the raw per-iteration series.

The summary statistics already hint at it: our shape-4 arm has sd 30.5% against
the reference's 5.2%, and our maximum (1102.6 us) lands within 1.2% of the
evaluator's shape-4 *best* (1115.8 us). If the series is bimodal with a slow
mode near 1.1 ms, then the evaluator run is simply our slow mode entered and
never left -- which would explain its 12% and 43% relative stdevs and retire
the whole "the evaluator disagrees" puzzle.

Reads whatever per-rank JSON the graded harness wrote; makes no assumptions
about the schema beyond "somewhere in here is a list of per-iteration numbers".
"""
import glob
import json
import os
import sys


def walk_numeric_lists(obj, path=""):
    """Yield (path, list-of-floats) for every numeric list in a JSON tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numeric_lists(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        if len(obj) >= 8 and all(isinstance(x, (int, float)) for x in obj):
            yield path, [float(x) for x in obj]
        else:
            for i, v in enumerate(obj):
                yield from walk_numeric_lists(v, f"{path}[{i}]")


def describe(name, xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = var ** 0.5

    def pct(p):
        return xs[min(n - 1, int(p * n))]

    print(f"\n--- {name}  (n={n}) ---")
    print(f"  min {xs[0]:10.1f}   p50 {pct(.50):10.1f}   p90 {pct(.90):10.1f}")
    print(f"  p95 {pct(.95):10.1f}   p99 {pct(.99):10.1f}   max {xs[-1]:10.1f}")
    print(f"  mean {mean:9.1f}   sd {sd:9.1f}  ({100*sd/mean:.1f}%)")

    # Bimodality test that needs no library: split at the midpoint between the
    # extremes and see whether both halves are populated with a real gap.
    lo, hi = xs[0], xs[-1]
    if hi <= lo * 1.05:
        print("  UNIMODAL (range < 5%)")
        return
    mid = (lo + hi) / 2
    fast = [x for x in xs if x <= mid]
    slow = [x for x in xs if x > mid]
    fm = sum(fast) / len(fast) if fast else 0.0
    sm = sum(slow) / len(slow) if slow else 0.0
    print(f"  split at {mid:.1f}:  fast n={len(fast)} mean={fm:.1f}   "
          f"slow n={len(slow)} mean={sm:.1f}")
    if fast and slow:
        gap = (min(slow) - max(fast)) / max(fast) * 100.0
        print(f"  gap between modes: {gap:.1f}% of the fast mode")
        if len(slow) >= max(2, 0.02 * n) and sm > 1.5 * fm:
            print(f"  ** BIMODAL: a slow mode at ~{sm:.0f} us, "
                  f"{100*len(slow)/n:.1f}% of iterations, {sm/fm:.2f}x the fast mode **")
        else:
            print("  looks like a heavy tail rather than a distinct mode")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        print(f"no json under {root}")
        return 1
    print(f"scanning {len(files)} files under {root}")
    for path in files:
        try:
            with open(path) as fh:
                blob = json.load(fh)
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        base = os.path.basename(path)
        for key, xs in walk_numeric_lists(blob):
            # Per-iteration latency series only: skip tiny/constant vectors.
            if len(xs) < 16:
                continue
            if max(xs) - min(xs) < 1e-9:
                continue
            describe(f"{base} :: {key}", xs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
