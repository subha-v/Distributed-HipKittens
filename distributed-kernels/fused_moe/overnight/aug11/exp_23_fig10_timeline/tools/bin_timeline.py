#!/usr/bin/env python3
"""exp_23 binner: rank0_events.json -> timeline_bins.csv (schema exp23-bins-1).

Usage:
  python3 bin_timeline.py --events rank0_events.json [rank0_events_mode0.json ...] \
      [--bin-us 10] [--bytes-model bytes_model.json] \
      --out timeline_bins.csv [--checks-out timeline_checks.json] \
      [--check-integrals]

Schema exp23-bins-1 (CSV, one header row, one row per (arm, bin))
-----------------------------------------------------------------
arm                 str    arm label, verbatim from the events file
bin_index           int    0-based, contiguous, from the arm's own origin
t_start_us          float  bin start, us from that arm's origin
t_end_us            float  bin end
grid_ctas           int    denominator for every fraction below (256)
ctas_dispatch       float  mean number of CTAs whose `dispatch` covers the bin
ctas_plan           float  ... `plan`
ctas_m6             float  ... `M6`
ctas_m7             float  ... `M7`
ctas_service        float  ... `service`
ctas_m75            float  ... `m75`
ctas_combine        float  ... `combine`
ctas_tail           float  ... `tail`
mfma_frac           float  (ctas_m6 + ctas_m7) / grid_ctas -- OCCUPANCY PROXY,
                           not MFMA utilisation. A CTA stalled on vmcnt inside
                           M7 counts here. Label it as such in the figure.
phase_coverage_frac float  fraction of the bin covered by ANY CTA's ANY phase,
                           divided by grid_ctas; < 1 means unstamped time
hbm_gbps            float  blank if the byte model has no number for a phase
xgmi_gbps           float  blank likewise
bytes_hbm           float  bytes attributed to this bin
bytes_xgmi          float
bytes_confidence    str    worst confidence among the phases active in the bin
                           (exact_from_shape < estimated < partial < absent)

Fractional CTA counts are intentional: a CTA whose interval covers 30 % of a
bin contributes 0.3, so the strip integrates to the true CTA-microseconds and
does not depend on the bin width. Byte rates use the same overlap weighting, so
sum(bytes over bins) == the model total EXACTLY (that is the integral check).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e23_lib as L  # noqa: E402

PHASES = [name for name, _a, _b in L.INTERVALS]
CONF_ORDER = ["exact_from_shape", "estimated", "partial", "absent"]


def _worse(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if CONF_ORDER.index(a) >= CONF_ORDER.index(b) else b


def bin_one_arm(doc, bin_us, model):
    """Returns (rows, checks) for one events document."""
    grid = doc["grid_ctas"]
    ivs = [(i["phase"], i["t0_us"], i["t1_us"])
           for c in doc["ctas"] for i in c["intervals"]]
    if not ivs:
        raise SystemExit(f"{doc['arm']}: no intervals to bin")
    t_lo = min(t0 for _p, t0, _t1 in ivs)
    t_hi = max(t1 for _p, _t0, t1 in ivs)
    nbins = max(1, int(math.ceil((t_hi - t_lo) / bin_us)))

    # overlap[phase][bin] = CTA-microseconds of that phase inside that bin
    overlap = {p: [0.0] * nbins for p in PHASES}
    total_us = {p: 0.0 for p in PHASES}
    for phase, t0, t1 in ivs:
        if t1 <= t0:
            continue
        total_us[phase] += t1 - t0
        b0 = int((t0 - t_lo) // bin_us)
        b1 = min(nbins - 1, int((t1 - t_lo) // bin_us))
        for b in range(b0, b1 + 1):
            lo = t_lo + b * bin_us
            hi = lo + bin_us
            ov = min(t1, hi) - max(t0, lo)
            if ov > 0:
                overlap[phase][b] += ov

    # bytes: uniform rate per CTA-microsecond within a phase, so the integral is
    # exact by construction (sum_b overlap * rate == total_us * rate == B).
    mp = (model or {}).get("phases", {})
    rate = {}          # phase -> (hbm B/us, xgmi B/us, confidence)
    for p in PHASES:
        entry = mp.get(p)
        if not entry or total_us[p] <= 0:
            rate[p] = (None, None, entry.get("confidence") if entry else "absent")
            continue
        hb, xb = entry.get("hbm_bytes"), entry.get("xgmi_bytes")
        rate[p] = (
            None if hb is None else hb / total_us[p],
            None if xb is None else xb / total_us[p],
            entry.get("confidence", "absent"),
        )

    rows = []
    for b in range(nbins):
        lo = t_lo + b * bin_us
        row = {
            "arm": doc["arm"],
            "bin_index": b,
            "t_start_us": round(lo, 4),
            "t_end_us": round(lo + bin_us, 4),
            "grid_ctas": grid,
        }
        cov = 0.0
        hbm = xgmi = 0.0
        hbm_known = xgmi_known = True
        conf = None
        for p in PHASES:
            ctas = overlap[p][b] / bin_us
            row["ctas_" + p.lower()] = round(ctas, 6)
            cov += overlap[p][b]
            if overlap[p][b] > 0:
                hr, xr, c = rate[p]
                conf = _worse(conf, c)
                if hr is None:
                    hbm_known = False
                else:
                    hbm += hr * overlap[p][b]
                if xr is None:
                    xgmi_known = False
                else:
                    xgmi += xr * overlap[p][b]
        row["mfma_frac"] = round(
            sum(overlap[p][b] for p in L.MFMA_INTERVALS) / bin_us / grid, 6)
        row["phase_coverage_frac"] = round(cov / bin_us / grid, 6)
        secs = bin_us * 1e-6
        row["bytes_hbm"] = round(hbm, 1) if hbm_known else ""
        row["bytes_xgmi"] = round(xgmi, 1) if xgmi_known else ""
        row["hbm_gbps"] = round(hbm / secs / 1e9, 4) if hbm_known else ""
        row["xgmi_gbps"] = round(xgmi / secs / 1e9, 4) if xgmi_known else ""
        row["bytes_confidence"] = conf or "absent"
        rows.append(row)

    # --- integral check: binned bytes must reproduce the model totals.
    integ = {"pass": True, "tolerance_frac": L.TOL_BIN_INTEGRAL_FRAC,
             "phases": []}
    for p in PHASES:
        entry = mp.get(p)
        if not entry:
            continue
        for kind, idx in (("hbm_bytes", 0), ("xgmi_bytes", 1)):
            want = entry.get(kind)
            if want is None or total_us[p] <= 0:
                continue
            got = sum(overlap[p][b] for b in range(nbins)) * (rate[p][idx] or 0.0)
            rel = 0.0 if want == 0 else abs(got - want) / abs(want)
            ok = rel <= L.TOL_BIN_INTEGRAL_FRAC or (want == 0 and got == 0)
            integ["pass"] = integ["pass"] and ok
            integ["phases"].append({"phase": p, "kind": kind, "model": want,
                                    "binned": got, "rel": rel, "pass": ok})

    # --- CTA-microsecond conservation: the strips must account for exactly the
    # CTA-microseconds the events file contains. Pure arithmetic; any failure is
    # a binning bug, which is why the selftest mutates it.
    src = sum(t1 - t0 for _p, t0, t1 in ivs)
    binned = sum(sum(overlap[p]) for p in PHASES)
    cons_rel = abs(binned - src) / src if src else 0.0
    conserve = {"pass": cons_rel <= 1e-9, "source_cta_us": src,
                "binned_cta_us": binned, "rel": cons_rel}

    # --- epoch averages, the quantities the amd-smi cross-check compares to.
    span_us = t_hi - t_lo
    avg = {"span_us": span_us,
           "mean_mfma_frac": (sum(sum(overlap[p]) for p in L.MFMA_INTERVALS)
                              / span_us / grid) if span_us else None}
    for kind, idx in (("hbm", 0), ("xgmi", 1)):
        tot = 0.0
        known = True
        for p in PHASES:
            e = mp.get(p) or {}
            v = e.get(kind + "_bytes")
            if v is None:
                if total_us[p] > 0:
                    known = False
                continue
            tot += v
        avg[kind + "_bytes_per_epoch"] = tot if known else None
        avg[kind + "_gbps_epoch_avg"] = (
            tot / (span_us * 1e-6) / 1e9 if known and span_us else None)

    return rows, {"arm": doc["arm"], "bin_us": bin_us, "nbins": nbins,
                  "integral": integ, "cta_us_conservation": conserve,
                  "epoch_average": avg,
                  "phase_cta_us": {p: total_us[p] for p in PHASES},
                  "events_verdict": doc.get("verdict")}


FIELDS = (["arm", "bin_index", "t_start_us", "t_end_us", "grid_ctas"]
          + ["ctas_" + p.lower() for p in PHASES]
          + ["mfma_frac", "phase_coverage_frac", "bytes_hbm", "bytes_xgmi",
             "hbm_gbps", "xgmi_gbps", "bytes_confidence"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--bin-us", type=float, default=10.0)
    ap.add_argument("--bytes-model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks-out", default=None)
    ap.add_argument("--check-integrals", action="store_true")
    a = ap.parse_args(argv)

    model = None
    if a.bytes_model:
        with open(a.bytes_model, "r", encoding="utf-8") as fh:
            model = json.load(fh)
        if model.get("schema") != "exp23-bytes-1":
            raise SystemExit("byte model schema mismatch")

    all_rows, all_checks = [], []
    for path in a.events:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("schema") != L.EVENTS_SCHEMA:
            raise SystemExit(f"{path}: schema {doc.get('schema')} != {L.EVENTS_SCHEMA}")
        rows, chk = bin_one_arm(doc, a.bin_us, model)
        all_rows += rows
        all_checks.append(chk)

    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    out = {"schema": L.BINS_SCHEMA, "bin_us": a.bin_us,
           "bytes_model": a.bytes_model, "arms": all_checks}
    if a.checks_out:
        with open(a.checks_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")

    bad = False
    for c in all_checks:
        print(f"[E23 BIN] {c['arm']}: {c['nbins']} bins of {c['bin_us']} us, "
              f"span {c['epoch_average']['span_us']:.1f} us, "
              f"mean_mfma_frac={c['epoch_average']['mean_mfma_frac']:.4f}")
        for k in ("hbm", "xgmi"):
            g = c["epoch_average"][k + "_gbps_epoch_avg"]
            print(f"[E23 BIN]   {k}_epoch_avg_GBps="
                  f"{'n/a (model incomplete)' if g is None else round(g, 3)}")
        if a.check_integrals:
            for name in ("integral", "cta_us_conservation"):
                ok = c[name]["pass"]
                bad = bad or not ok
                print(f"[E23 CHECK] {c['arm']} {name:22s} "
                      f"{'PASS' if ok else 'FAIL'}")
            if not c["integral"]["pass"]:
                for r in c["integral"]["phases"]:
                    if not r["pass"]:
                        print(f"[E23 CHECK]   {r['phase']}/{r['kind']}: "
                              f"model={r['model']} binned={r['binned']:.1f} "
                              f"rel={r['rel']:.4g}")
    return 3 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
