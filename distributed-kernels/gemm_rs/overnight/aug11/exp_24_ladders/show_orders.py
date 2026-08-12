#!/usr/bin/env python3
"""Print the arm orders one shape actually walked, plus the pairwise-offset
diagnostic that decides whether the ordering is defective.

For each ordered pair (j, k) of arms it reports how many distinct relative
offsets that pair took across the reps. Under the inherited cyclic scheme the
answer is 1 for every pair -- that IS the defect: arm j always ran the same
number of slots after arm k, so a neighbour effect became a constant offset on
that pair. Under a shuffle the answer should be several.
"""
import collections
import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    orders = d.get("arm_orders") or []
    if not orders:
        print(f"  shape {d['shape_index'] + 1}: no arm_orders recorded "
              f"(pre-fix sample file)")
        return 0
    arms = list(orders[0])
    pos = [{a: o.index(a) for a in arms} for o in orders]
    offsets = collections.defaultdict(set)
    for p in pos:
        for j in arms:
            for k in arms:
                if j != k:
                    offsets[(j, k)].add((p[j] - p[k]) % len(arms))
    distinct = [len(v) for v in offsets.values()]
    pinned = [f"{j}->{k}" for (j, k), v in offsets.items() if len(v) == 1]
    print(f"  shape {d['shape_index'] + 1} {d['shape_label']:>16} "
          f"mode={d.get('rot_mode')} reps={d['reps']}")
    print(f"      first-position arms : {[o[0] for o in orders]}")
    print(f"      distinct relative offsets per arm pair: "
          f"min={min(distinct)} max={max(distinct)} "
          f"(1 == pinned pair == the defect)")
    if pinned:
        print(f"      PINNED PAIRS ({len(pinned)}): {pinned[:6]}")
    else:
        print(f"      no pinned pairs -- every pair's spacing varies across reps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
