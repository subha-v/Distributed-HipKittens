#!/usr/bin/env python3
"""exp_22: timeline_bins.csv -> the Fig 3 grid.

Rows are the three resources, columns are the arms, x is shared (us from epoch
start) and y is shared per row so the columns are visually comparable -- the
sibling's 3xN layout, kept unchanged so the two operator instances read the
same way (FIGURE_SPECS.md section 6).

The MFMA row is labelled "CTAs in MFMA phase (occupancy proxy)", not MFMA
utilization, and the denominator is printed in the axis label so no reader has
to go and find it.

  plot_timeline.py timeline_bins.csv --validation validation.json -o fig3.png
"""

import argparse
import csv
import json
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

ARM_TITLES = {
    "b0_reference": "(a) reference GEMM + RCCL",
    "ours": "(b) ours — GEMM-RS megakernel",
    "rank1": "(c) frozen rank-1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--validation", default=None)
    ap.add_argument("-o", "--out", default="fig3_timeline.png")
    ap.add_argument("--title", default="Per-layer resource utilization, "
                                       "8x MI300X, shape 8192x4096x14336")
    args = ap.parse_args()

    arms = OrderedDict()
    with open(args.csv, newline="") as handle:
        for row in csv.DictReader(handle):
            arms.setdefault(row["arm"], []).append(row)
    order = [a for a in ("b0_reference", "ours", "rank1") if a in arms]
    order += [a for a in arms if a not in order]

    validation = {}
    if args.validation:
        with open(args.validation) as handle:
            validation = json.load(handle).get("arms", {})

    rows = [
        ("mfma_frac", "CTAs in MFMA phase\n(occupancy proxy)", None),
        ("hbm_gbs", "HBM GB/s\n(bytes issued)", None),
        ("xgmi_gbs", "xGMI GB/s\n(peer egress)", None),
    ]
    fig, axes = plt.subplots(len(rows), len(order), sharex="col",
                             figsize=(5.2 * len(order), 8.4), squeeze=False)

    for ri, (field, label, _) in enumerate(rows):
        top = max(float(r[field]) for arm in order for r in arms[arm]) or 1.0
        for ci, arm in enumerate(order):
            ax = axes[ri][ci]
            data = arms[arm]
            xs = [float(r["t_us_start"]) for r in data]
            ys = [float(r[field]) for r in data]
            ax.fill_between(xs, ys, step="post", alpha=0.45)
            ax.step(xs, ys, where="post", linewidth=1.0)
            ax.set_ylim(0, top * 1.08)
            ax.grid(alpha=0.25, linewidth=0.5)
            if ri == 0:
                denom = data[0].get("mfma_denominator", "?")
                ax.set_title(f"{ARM_TITLES.get(arm, arm)}\n"
                             f"denominator = {denom} CTAs", fontsize=10)
            if ri == len(rows) - 1:
                ax.set_xlabel("us from epoch start")
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)

            report = validation.get(arm, {}).get("integrals", {})
            if ri == 2 and report:
                flags = " ".join(
                    f"{k.split('_')[0]}:{'ok' if v['pass'] else 'FAIL'}"
                    for k, v in report.items())
                ax.annotate(f"integral {flags}", xy=(0.02, 0.88),
                            xycoords="axes fraction", fontsize=7.5)

    fig.suptitle(args.title + "\nrank 0; instrumented build is a diagnostic "
                              "arm and its us are not performance numbers",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
