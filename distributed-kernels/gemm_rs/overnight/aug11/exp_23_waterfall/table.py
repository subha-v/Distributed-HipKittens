"""Render the exp_23 waterfall as markdown tables for result.md.

Reads stats.json (pooled, widened-null scoring) and prints the two tables the
figure needs: rung x shape with best AND median, and the NR sweep with the
shipped point marked. No new computation -- purely a view over stats.json, so
the numbers in result.md cannot drift from the ones the scorer produced.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = ["a", "b", "c", "null"]
NRS = ["nr8", "nr16", "nr32", "nr48"]


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    data = json.load(open(os.path.join(HERE, "stats.json")))
    shapes = data["shapes"]
    order = sorted(shapes, key=int)

    print(f"draws: {len(data['draws'])} "
          f"({len(data['draws']) - data['draws_reversed']} forward, "
          f"{data['draws_reversed']} reversed construction)\n")

    print("### Waterfall: rung x shape, best / median us\n")
    hdr = "| rung | " + " | ".join(
        f"{shapes[i]['shape']}" for i in order) + " | geomean |"
    print(hdr)
    print("|" + "---|" * (len(order) + 2))
    for rung in RUNGS:
        cells, bests, meds = [], [], []
        for i in order:
            arm = shapes[i]["arms"].get(rung, {})
            if not arm.get("median_us"):
                cells.append("dead")
                continue
            bests.append(arm["best_us"])
            meds.append(arm["median_us"])
            cells.append(f"{arm['best_us']:.1f} / {arm['median_us']:.1f}")
        geo = (f"{geomean(bests):.2f} / {geomean(meds):.2f}"
               if len(meds) == len(order) else "-")
        print(f"| {rung} | " + " | ".join(cells) + f" | **{geo}** |")

    print("\n### Rung deltas (median gain % vs previous rung, "
          "widened null floor, verdict)\n")
    print("| shape | mechanism active | null floor % | a->b | b->c |")
    print("|---|---|---|---|---|")
    for i in order:
        sh = shapes[i]
        row = [sh["shape"],
               f"a->b {sh['ab_rung_active']}, b->c {sh['bc_rung_active']}",
               f"{sh['null_floor_worst_draw_pct']:.2f} "
               f"(1-pair: {sh['null_floor_single_pair_pct']:.2f})"]
        for name in ("b vs a", "c vs b"):
            c = sh["contrasts"][name]
            row.append(f"{c['gain_median_pct']:+.2f}% "
                       f"[{c['gain_median_range'][0]:+.2f},"
                       f"{c['gain_median_range'][1]:+.2f}] "
                       f"p={min(c['p_ranksum_faster'], c['p_ranksum_slower']):.4f} "
                       f"**{c['verdict']}**")
        print("| " + " | ".join(row) + " |")

    print("\n### NR sweep at the rung-c config (median gain % vs shipped NR)\n")
    print("| shape | shipped NR | NR=8 | NR=16 | NR=32 | NR=48 |")
    print("|---|---|---|---|---|---|")
    for i in order:
        sh = shapes[i]
        row = [sh["shape"], str(sh["shipped_nr"])]
        for nr in NRS:
            key = next(k for k in sh["contrasts"] if k.startswith(nr + " vs"))
            c = sh["contrasts"][key]
            mark = " **(=shipped, null pair)**" if "*shipped" in key else ""
            verdict = ("null" if c.get("is_null_pair")
                       else "resolved slower" if c["verdict"].endswith("slower")
                       else "resolved faster" if c["verdict"].endswith("faster")
                       else "unresolved")
            row.append(f"{c['gain_median_pct']:+.2f}% {verdict}{mark}")
        print("| " + " | ".join(row) + " |")

    print("\n### NR sweep geomeans\n")
    print("| arm | geomean best us | geomean median us | vs rung c |")
    print("|---|---|---|---|")
    geo = data["geomeans"]
    cmed = geo["c"]["geomean_median_us"]
    for label in RUNGS + NRS:
        g = geo.get(label)
        if not g:
            continue
        print(f"| {label} | {g['geomean_best_us']:.2f} | "
              f"{g['geomean_median_us']:.2f} | "
              f"{g['geomean_median_us'] / cmed:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
