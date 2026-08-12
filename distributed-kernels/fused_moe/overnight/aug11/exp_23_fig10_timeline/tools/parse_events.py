#!/usr/bin/env python3
"""exp_23 parser: run log -> rank0_events.json (schema exp23-events-1).

Usage:
  python3 parse_events.py --log run1.log --arm mps_mega \
      --cfg "C=16,g=353,mode=12,flush_rows=16,timestamps=1" \
      [--summary summary.json] [--head <sha>] [--src-rev 28] [--tier A+B] \
      --out rank0_events.json

Exit codes: 0 = all applicable checks passed; 3 = a check FAILED (the JSON is
still written, so the failure is inspectable); 2 = the log could not be parsed.

Schema exp23-events-1
---------------------
schema        str    "exp23-events-1"
arm           str    harness arm name, or "mps_mega:mode0" for the bulk panel
cfg           str    the K0_MPS_CFG string that produced it
head          str    node checkout short SHA (provenance, may be "")
src_rev       int    K0P6_MPS_SRC_REV (provenance, may be null)
tier          str    which instrument tier was compiled ("A", "A+B", "A+B+C")
rank          int    always 0 (the harness prints only rank 0)
tick_ns       int    10
tick_us       float  0.01
grid_ctas     int    256
slots         int    16
phase_slots   {int:str}  slot index -> boundary name
intervals_def [[name, open, close]]  interval definition actually used
epoch_window  {lo_ticks, max_ticks, stale_cells}
origin_ticks  int    x-axis origin = min KSTART (or min M2_DONE if absent)
ctas          [ {bid, role, meta_count,
                 stamps:{name: ticks}, stamps_us:{name: us_from_origin},
                 intervals:[{phase, t0_us, t1_us, dur_us}]} ]
coarse        {name: ticks}      the [MPS TS] cells, verbatim
spin          {success_max, fail_max} or null
e2e           {arm_p50_us, ...}
checks        {monotonic, coarse_reconcile, coverage, interior, e2e}
verdict       {"pass": bool, "failed": [check names]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e23_lib as L  # noqa: E402


def build(log_text, arm, cfg, arm_p50_us=None, head="", src_rev=None,
          tier="A+B"):
    p = L.parse_log(log_text)
    lo, t_max, stale = L.epoch_window(p["stamps"], p["coarse"])
    ctas = L.build_ctas(p["stamps"], lo)

    starts = [c["stamps"]["KSTART"] for c in ctas if "KSTART" in c["stamps"]]
    if starts:
        origin = min(starts)
    else:
        m2 = [c["stamps"]["M2_DONE"] for c in ctas if "M2_DONE" in c["stamps"]]
        if not m2:
            raise L.ParseError("neither KSTART nor M2_DONE present: no origin")
        origin = min(m2)

    for c in ctas:
        c["stamps_us"] = {k: (v - origin) * L.TICK_US
                          for k, v in c["stamps"].items()}
        c["intervals"] = [
            {"phase": i["phase"],
             "t0_us": (i["t0"] - origin) * L.TICK_US,
             "t1_us": (i["t1"] - origin) * L.TICK_US,
             "dur_us": (i["t1"] - i["t0"]) * L.TICK_US}
            for i in c["intervals"]
        ]

    checks = {
        "monotonic": L.check_monotonic(ctas),
        "coarse_reconcile": L.check_coarse(ctas, p["coarse"]),
        "coverage": L.check_coverage(
            # coverage needs raw ticks; recompute from stamps
            [{"bid": c["bid"],
              "intervals": [{"t0": int(round(i["t0_us"] / L.TICK_US)),
                             "t1": int(round(i["t1_us"] / L.TICK_US))}
                            for i in c["intervals"]]} for c in ctas]),
        "interior": L.check_interior(ctas, p["coarse"]),
        "e2e": L.check_e2e(ctas, arm_p50_us),
    }
    failed = [k for k, v in checks.items() if v.get("pass") is False]
    if stale:
        failed.append("epoch_window")

    return {
        "schema": L.EVENTS_SCHEMA,
        "arm": arm,
        "cfg": cfg,
        "head": head,
        "src_rev": src_rev,
        "tier": tier,
        "rank": 0,
        "tick_ns": p["tick_ns"],
        "tick_us": L.TICK_US,
        "grid_ctas": p["n_ctas_declared"],
        "slots": L.SLOTS,
        "phase_slots": {str(k): v for k, v in L.PHASE_SLOTS.items()},
        "intervals_def": [list(t) for t in L.INTERVALS],
        "mfma_intervals": list(L.MFMA_INTERVALS),
        "epoch_window": {"lo_ticks": lo, "max_ticks": t_max,
                         "stale_cells": stale},
        "origin_ticks": origin,
        "ctas": ctas,
        "coarse": p["coarse"],
        "spin": p["spin"],
        "e2e": {"arm_p50_us": arm_p50_us,
                "source": "summary.json arm_p50_us[arm].median"},
        "checks": checks,
        "verdict": {"pass": not failed, "failed": failed},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--head", default="")
    ap.add_argument("--src-rev", type=int, default=None)
    ap.add_argument("--tier", default="A+B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with open(a.log, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    p50 = L.read_arm_p50(a.summary, a.arm) if a.summary else None

    try:
        doc = build(text, a.arm, a.cfg, p50, a.head, a.src_rev, a.tier)
    except L.ParseError as e:
        print(f"[E23 PARSE] FAILED: {e}", file=sys.stderr)
        return 2

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")

    v = doc["verdict"]
    print(f"[E23 PARSE] {a.arm}: {len(doc['ctas'])} CTAs, "
          f"origin={doc['origin_ticks']}, stale={doc['epoch_window']['stale_cells']}")
    for name, chk in doc["checks"].items():
        st = {True: "PASS", False: "FAIL", None: "N/A"}[chk.get("pass")]
        print(f"[E23 CHECK] {name:18s} {st}")
    print(f"[E23 PARSE] verdict={'PASS' if v['pass'] else 'FAIL ' + ','.join(v['failed'])}")
    return 0 if v["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
