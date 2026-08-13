#!/usr/bin/env python3
"""Parse exp_01's raw/ tree of popcorn outputs into per-arm tables.

Usage: parse_popcorn.py <raw_dir>

Reads every raw/<session>/<arm>/<label>.popcorn.txt, refuses files without
`check: pass` or with fewer than 6 benchmark blocks (the exp_24 parser lesson:
a truncated evaluator run still emits `benchmark-count: 6`), and prints
per-shape best and mean (us) plus geomeans, then within-session ratios.
"""
import json
import math
import re
import sys
from pathlib import Path

SHAPES = ["64", "512", "2048", "4096", "8192a", "8192b"]


def parse_file(path):
    text = path.read_text()
    if "check: pass" not in text:
        return None, f"no 'check: pass' in {path}"
    fields = {}
    for line in text.splitlines():
        m = re.match(r"benchmark\.(\d+)\.(\w+): (.*)", line)
        if m:
            idx, key, val = int(m.group(1)), m.group(2), m.group(3)
            fields.setdefault(idx, {})[key] = val
    if len(fields) < 6:
        return None, f"only {len(fields)} benchmark blocks in {path}"
    best = [float(fields[i]["best"]) / 1000.0 for i in range(6)]   # ns -> us
    mean = [float(fields[i]["mean"]) / 1000.0 for i in range(6)]
    return {"best": best, "mean": mean}, None


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main(raw):
    raw = Path(raw)
    rows = {}          # (session, arm) -> parsed
    for pop in sorted(raw.glob("*/*/*.popcorn.txt")):
        session, arm = pop.parent.parent.name, pop.parent.name
        if arm.endswith("_test") or "test" in pop.name:
            continue
        parsed, err = parse_file(pop)
        if err:
            print(f"SKIP {err}")
            continue
        rows[(session, arm)] = parsed

    print(f"\n{'sess':>5} {'arm':>22} " +
          " ".join(f"{s:>8}" for s in SHAPES) + f" {'GM(best)':>9} {'GM(mean)':>9}")
    out = {}
    for (session, arm), p in sorted(rows.items()):
        gb, gm = geomean(p["best"]), geomean(p["mean"])
        out[f"{session}/{arm}"] = {"best_us": p["best"], "mean_us": p["mean"],
                                   "gm_best": gb, "gm_mean": gm}
        print(f"{session:>5} {arm:>22} " +
              " ".join(f"{v:8.1f}" for v in p["best"]) + f" {gb:9.1f} {gm:9.1f}")

    print("\nWithin-session ratios (geomean of per-shape best):")
    for session in sorted({s for s, _ in rows}):
        arms = {a: p for (s, a), p in rows.items() if s == session}
        if "O" in arms:
            o = geomean(arms["O"]["best"])
            for other in ("K", "R", "D"):
                if other in arms:
                    r = geomean(arms[other]["best"])
                    print(f"  {session}: O/{other} = {o / r:.4f}   "
                          f"(O {o:.1f} vs {other} {r:.1f})")
    (Path(raw) / "parsed.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {raw}/parsed.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw")
