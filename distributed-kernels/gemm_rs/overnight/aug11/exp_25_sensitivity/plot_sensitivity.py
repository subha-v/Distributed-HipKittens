#!/usr/bin/env python3
"""exp_25 plots. Reads knob_by_shape.json only; no GPU, no measurement.

Panel A (2x2): rung delta vs communication share, one subplot per comm-share
definition, so the reader sees that the sign of the trend does not depend on the
definition. Structurally inert shapes are drawn hollow — they are arithmetic,
not evidence — and each point carries its shape's own widened floor as the
error bar, so a delta inside its floor is visibly inside it.

Panel B: the NR curve per shape against each shape's widened floor band, with
the shipped NR marked. The plateau/cliff structure is the Q2 exhibit.

Not executed on the authoring host (no matplotlib there); run it wherever the
figures are rendered.

    python3 plot_sensitivity.py            # writes both PNGs next to the JSON
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFS = [
    ("D1_egress_over_full", "D1  egress / full"),
    ("D2_egress_sync_release_over_full", "D2  (egress+sync+release) / full"),
    ("D3_full_minus_gemm_over_full", "D3  (full - GEMM) / full   [primary]"),
    ("D4_comm_over_priced_pools", "D4  comm / sum(priced pools)"),
]
RUNGS = [("a_to_b", "a->b  task order (WGM)", "tab:blue", "o"),
         ("b_to_c", "b->c  release granularity", "tab:orange", "s")]


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required to render these panels; the data in "
                 "knob_by_shape.json / sensitivity_points.csv is complete "
                 "without it.")

    d = json.load(open(os.path.join(HERE, "knob_by_shape.json"),
                       encoding="utf-8"))
    usable = [s for s in d["shapes"] if s["comm_share"]["usable_for_correlation"]]
    excluded = [s for s in d["shapes"]
                if not s["comm_share"]["usable_for_correlation"]]

    # ---------------------------------------------------------------- panel A --
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for ax, (key, label) in zip(axes.ravel(), DEFS):
        rho = d["correlations"]["by_definition"][key]["a_to_b_mask_variant"][
            "spearman_rho"]
        for rung, rlabel, colour, marker in RUNGS:
            for s in usable:
                r = s["rungs"][rung]
                x = s["comm_share"][key]
                y = r["gain_median_pct"]
                filled = r["structurally_active"]
                ax.errorbar(
                    x, y, yerr=s["floors"]["exp23_widened_floor_pct"],
                    fmt=marker, color=colour, ms=8, capsize=3, elinewidth=1,
                    markerfacecolor=colour if filled else "none",
                    label=None, zorder=3)
                ax.annotate(" s%d" % s["shape_index"], (x, y), fontsize=8,
                            va="center")
        ax.axhline(0, color="0.7", lw=0.8, zorder=1)
        ax.set_title("%s\nSpearman rho (a->b, mask variant, n=4) = %+.2f"
                     % (label, rho), fontsize=9)
        ax.set_xlabel("communication share at rung c")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("rung gain, % (positive = later rung faster)")
    axes[1][0].set_ylabel("rung gain, % (positive = later rung faster)")
    handles = [
        plt.Line2D([], [], color="tab:blue", marker="o", ls="",
                   label="a->b task order, structurally ACTIVE"),
        plt.Line2D([], [], color="tab:blue", marker="o", ls="",
                   markerfacecolor="none", label="a->b, structurally INERT"),
        plt.Line2D([], [], color="tab:orange", marker="s", ls="",
                   label="b->c granularity, ACTIVE"),
        plt.Line2D([], [], color="tab:orange", marker="s", ls="",
                   markerfacecolor="none", label="b->c, INERT"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9)
    fig.suptitle("exp_25 / Q5 — per-shape rung delta vs communication share.\n"
                 "Hollow = the knob is arithmetically inert on that shape "
                 "(zero is not a measurement). Bars = that shape's own widened "
                 "null floor. Shapes %s excluded: HOST-bound."
                 % ", ".join("s%d" % s["shape_index"] for s in excluded),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    a_path = os.path.join(HERE, "fig_q5_delta_vs_commshare.png")
    fig.savefig(a_path, dpi=160)

    # ---------------------------------------------------------------- panel B --
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(d["shapes"]):
        pts = sorted(s["nr_curve"]["points"], key=lambda p: p["nr"])
        xs = [p["nr"] for p in pts]
        ys = [p["gain_median_pct_vs_shipped"] for p in pts]
        colour = cmap(i / max(1, len(d["shapes"]) - 1))
        ax.plot(xs, ys, "-o", color=colour, ms=5,
                label="s%d  %s (shipped NR=%d)"
                      % (s["shape_index"], s["shape"],
                         s["nr_curve"]["shipped_nr"]))
        for p in pts:
            if p["is_shipped_config"]:
                ax.plot(p["nr"], p["gain_median_pct_vs_shipped"], "*",
                        color=colour, ms=16, zorder=4)
        f = s["floors"]["exp23_widened_floor_pct"]
        ax.axhspan(-f, f, color=colour, alpha=0.05, zorder=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks([8, 16, 32, 48])
    ax.set_xlabel("NUM_REDUCER_CTAS (of 304)")
    ax.set_ylabel("gain vs that shape's shipped NR, % (negative = slower)")
    ax.set_title("exp_25 / Q2 — reducer-count curve. Star = shipped config.\n"
                 "Flat from NR=32 upward on shapes 2-6 (inside each shape's "
                 "widened floor); cliff below.")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    b_path = os.path.join(HERE, "fig_q2_nr_curve.png")
    fig2.savefig(b_path, dpi=160)

    print("wrote", a_path)
    print("wrote", b_path)


if __name__ == "__main__":
    main()
