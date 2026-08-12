#!/usr/bin/env python3
"""Dump one line per measured point with its roofline fraction, so an
implausible number is visible before the rest of the grid is spent."""
import json
import sys

PEAK_XGMI_LINK = 76.8      # GB/s, one MI350X xGMI link, unidirectional
PEAK_XGMI_AGG = 537.6      # GB/s, 7 links
PEAK_HBM = 8000.0          # GB/s
PEAK_MFMA = 2300.0         # TFLOPS, dense bf16


def peak_for(r):
    if r["mode"] == "xgmi":
        return PEAK_XGMI_AGG if r.get("fanout") == "rr7" else PEAK_XGMI_LINK
    if r["mode"] == "hbm":
        return PEAK_HBM
    return PEAK_MFMA


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: (r["mode"], str(r.get("fanout")), r.get("mlp") or 0,
                             r["concurrency"], r["ctas"]))
    hdr = ("mode   fan     mlp conc          C     value  metric   %peak  "
           "wall_us   ovl%  res_us  cmp_us iqr%")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pk = peak_for(r)
        frac = 100.0 * r["value"] / pk
        flag = "  <<< ABOVE PEAK" if frac > 100.5 else ""
        ovl = r.get("overlap_pct_median")
        print("%-6s %-7s %-3s %-13s %-4s %9.2f %-7s %6.1f %8.0f %6s %7s %7s %4.2f%s" % (
            r["mode"], r.get("fanout") or "-", r.get("mlp") or "-",
            r["concurrency"], r["ctas"], r["value"], r["metric"], frac,
            r.get("wall_us_median", 0.0),
            "-" if ovl is None else "%.1f" % ovl,
            "%.0f" % (r.get("res_span_us_median") or 0.0),
            "%.0f" % (r.get("cmp_span_us_median") or 0.0),
            r.get("spread", {}).get("rel_iqr_pct", 0.0), flag))
    print()
    print("points=%d" % len(rows))
    print("above-peak points: %d" % len([r for r in rows
                                         if 100.0 * r["value"] / peak_for(r) > 100.5]))
    seq = [r for r in rows if r["concurrency"] == "concurrent"
           and (r.get("overlap_pct_median") or 0.0) < 50.0]
    print("concurrent points with overlap<50%% (fake concurrency): %d" % len(seq))
    for r in seq:
        print("   ", r["key"], r.get("overlap_pct_median"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "saturation_quick.jsonl")
