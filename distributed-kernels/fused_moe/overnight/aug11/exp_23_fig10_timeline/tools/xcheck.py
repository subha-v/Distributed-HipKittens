#!/usr/bin/env python3
"""exp_23 external cross-check: counters + fabric ceiling vs the timeline.

MEASURED CONSTRAINT (probed on gbt350-odcdh2-c05-1, 2026-08-12, amd-smi 26.2.2
/ ROCm 7.2.4): there is NO live xGMI throughput counter on this node.
  * `amd-smi metric -g N --xgmi --csv` returns only `gpu,xgmi_err` -> `0,N/A`.
  * `rocm-smi --shownodesbw` prints `0-0 mps` for every pair; it is a topology
    capability field, not a counter.
  * The only live utilisation fields in `amd-smi metric --csv` are percentages:
    `gfx_activity`, `umc_activity` (plus PCIe `current_bandwidth_sent/received`,
    which is the host link, not the fabric).
  * Per-sample cost is 0.14-0.29 s, so the honest ceiling sampling rate is 3 Hz.

So plan.md's "amd-smi metric per-link xGMI throughput matches the xGMI strip
(+/-20 %)" is NOT PERFORMABLE as written. This tool implements what is:

  X1  HBM occupancy check   -- `umc_activity` duty cycle over the soak window vs
                               the timeline's time-weighted fraction of the epoch
                               spent in HBM-bearing phases. Both are fractions;
                               tolerance +/-15 percentage points.
  X2  window negative ctrl  -- `gfx_activity` must be >= 95 % over the window.
                               It SATURATES, so it cannot validate the MFMA
                               strip; it only proves the window contains no idle
                               time, i.e. that the timeline framing is sound.
  X3  fabric ceiling check  -- every xGMI GB/s the timeline implies must sit
                               below the measured fabric ceiling from aug10
                               exp_21 (54.9 GB/s for 16 B stores, 52.8 GB/s for
                               coalesced 4 B atomics -- mode 12's payload is the
                               atomic form, so 52.8 is the applicable bound).
                               A strip implying more than that is falsified on
                               its face. Weak, free, and real.

X4 (NOT implemented here, specified in result.md): rocprofv3 TCC EA read/write
size counters on one launch, to turn X3 from a ceiling into a measurement. That
needs a GPU slot and a counter-availability probe on gfx950.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e23_lib as L  # noqa: E402

# aug10 exp_21 fabric ubench, quoted in PAPER.md section 4.2 / PLOTS.md.
FABRIC_CEILING_GBPS = {"atomic_4B_coalesced": 52.8, "store_16B": 54.9}
TOL_UMC_ABS = 0.15          # percentage points / 100
GFX_ACTIVITY_FLOOR = 0.95

# Phases that carry HBM traffic at all. Used for the X1 duty-cycle comparison,
# which is about "is the memory system busy when we say it is", not GB/s.
HBM_PHASES = ("dispatch", "plan", "M6", "M7", "combine")


def parse_samples(path):
    """Read the sampler's CSV: utc_epoch_s,gpu,gfx_activity,umc_activity."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "t": float(r["utc_epoch_s"]),
                    "gpu": int(r["gpu"]),
                    "gfx": float(r["gfx_activity"]),
                    "umc": float(r["umc_activity"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"{path}: no usable samples "
                         "(expected utc_epoch_s,gpu,gfx_activity,umc_activity)")
    return rows


def timeline_fractions(doc):
    """Time-weighted fraction of the epoch each phase occupies, per CTA-average."""
    grid = doc["grid_ctas"]
    ivs = [(i["phase"], i["t0_us"], i["t1_us"])
           for c in doc["ctas"] for i in c["intervals"]]
    lo = min(t0 for _p, t0, _t1 in ivs)
    hi = max(t1 for _p, _t0, t1 in ivs)
    span = hi - lo
    per = {}
    for p, t0, t1 in ivs:
        per[p] = per.get(p, 0.0) + (t1 - t0)
    return span, {p: v / span / grid for p, v in per.items()}


def run(events_path, samples_path, window, gpu, peak_xgmi_gbps):
    with open(events_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("schema") != L.EVENTS_SCHEMA:
        raise SystemExit("events schema mismatch")
    span_us, frac = timeline_fractions(doc)
    hbm_frac = sum(frac.get(p, 0.0) for p in HBM_PHASES)

    out = {"schema": "exp23-xcheck-1", "arm": doc["arm"],
           "epoch_span_us": span_us, "phase_fractions": frac,
           "timeline_hbm_phase_fraction": hbm_frac, "checks": {}}

    if samples_path:
        s = [r for r in parse_samples(samples_path) if r["gpu"] == gpu]
        if window:
            s = [r for r in s if window[0] <= r["t"] <= window[1]]
        if not s:
            raise SystemExit("no samples for gpu/window")
        gfx = sum(r["gfx"] for r in s) / len(s) / 100.0
        umc = sum(r["umc"] for r in s) / len(s) / 100.0
        out["samples"] = {"n": len(s), "gpu": gpu,
                          "window_s": [s[0]["t"], s[-1]["t"]],
                          "duration_s": s[-1]["t"] - s[0]["t"],
                          "rate_hz": (len(s) - 1) / (s[-1]["t"] - s[0]["t"])
                                     if s[-1]["t"] > s[0]["t"] else None,
                          "mean_gfx_activity": gfx, "mean_umc_activity": umc}
        d = abs(umc - hbm_frac)
        out["checks"]["X1_hbm_occupancy"] = {
            "pass": d <= TOL_UMC_ABS, "umc_activity": umc,
            "timeline_hbm_fraction": hbm_frac, "abs_delta": d,
            "tolerance_abs": TOL_UMC_ABS,
            "note": "both are fractions of wall time, not GB/s"}
        out["checks"]["X2_window_not_idle"] = {
            "pass": gfx >= GFX_ACTIVITY_FLOOR, "mean_gfx_activity": gfx,
            "floor": GFX_ACTIVITY_FLOOR,
            "note": "negative control only; gfx_activity saturates and cannot "
                    "validate the MFMA strip"}
    else:
        out["checks"]["X1_hbm_occupancy"] = {"pass": None,
                                             "reason": "no --samples"}
        out["checks"]["X2_window_not_idle"] = {"pass": None,
                                               "reason": "no --samples"}

    ceiling = FABRIC_CEILING_GBPS["atomic_4B_coalesced"]
    if peak_xgmi_gbps is None:
        out["checks"]["X3_fabric_ceiling"] = {
            "pass": None, "reason": "no --peak-xgmi-gbps (byte model incomplete)",
            "ceiling_gbps": ceiling}
    else:
        out["checks"]["X3_fabric_ceiling"] = {
            "pass": peak_xgmi_gbps <= ceiling, "peak_xgmi_gbps": peak_xgmi_gbps,
            "ceiling_gbps": ceiling,
            "ceiling_provenance": "aug10 exp_21 fabric ubench, coalesced 4 B "
                                  "atomics; mode 12's payload is that form"}
    out["verdict"] = {
        "pass": all(v.get("pass") is not False for v in out["checks"].values()),
        "failed": [k for k, v in out["checks"].items() if v.get("pass") is False],
        "not_evaluated": [k for k, v in out["checks"].items()
                          if v.get("pass") is None],
    }
    return out


def peak_from_bins(bins_csv, arm):
    peak = None
    with open(bins_csv, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != arm or not r["xgmi_gbps"]:
                continue
            v = float(r["xgmi_gbps"])
            peak = v if peak is None else max(peak, v)
    return peak


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--bins", default=None,
                    help="timeline_bins.csv, to take the peak xGMI GB/s from")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--window", nargs=2, type=float, default=None,
                    metavar=("START_S", "END_S"))
    ap.add_argument("--peak-xgmi-gbps", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    peak = a.peak_xgmi_gbps
    if peak is None and a.bins:
        with open(a.events, "r", encoding="utf-8") as fh:
            arm = json.load(fh)["arm"]
        peak = peak_from_bins(a.bins, arm)

    out = run(a.events, a.samples, a.window, a.gpu, peak)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    for k, v in out["checks"].items():
        st = {True: "PASS", False: "FAIL", None: "N/A"}[v.get("pass")]
        print(f"[E23 XCHECK] {k:22s} {st}"
              + ("" if v.get("pass") is not None else f"  ({v.get('reason')})"))
    return 3 if out["verdict"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
