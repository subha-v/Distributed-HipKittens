#!/usr/bin/env python3
"""exp_22: turn saturation_<tier>.jsonl into plot-ready saturation.json.

Usage: summarize.py IN.jsonl OUT.json [--print]

Everything derived here is mechanical -- including the H1-H4 verdicts, so the
pre-registered hypotheses are adjudicated by code against thresholds fixed
before the data existed, not by prose after it.

Knee definition (same one plan.md's plot spec asks for): the smallest CTA count
whose value reaches 90% of that series' plateau, where the plateau is the median
of the top two CTA counts in the series. Reported with the value at the knee and
the plateau, so a series that never plateaus is visible rather than silently
knee-annotated.
"""
import json
import statistics as st
import sys

SCHEMA = "exp22-saturation-1"

# Pre-registered thresholds (plan.md). Frozen: do not tune these to the data.
H1_MAX_CTAS = 8          # single-link push saturates by 8 CTAs at MLP >= 4
H1_FRAC = 0.75           # "saturates" = >= 75% of 76.8 GB/s
H2_MAX_CTAS = 32         # aggregate egress saturates by 16-32 CTAs
H3_DEPRESSION = 0.90     # concurrent < 90% of isolated counts as "depressed"
H4_R2 = 0.98             # TFLOPS vs CTAs linear
FALSIFIER_CTAS = 32      # >= 32 CTAs needed for 75% of a single link inverts
                         # the "tiny topology-sized pool" story


def load(path):
    pts = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                pts.append(json.loads(ln))
            except json.JSONDecodeError:
                print("WARN: unparsable line skipped (truncated by a kill?)",
                      file=sys.stderr)
    return pts


def series(pts, **sel):
    out = [p for p in pts if all(p.get(k) == v for k, v in sel.items())]
    return sorted(out, key=lambda p: p["ctas"])


def knee(s, frac=0.90):
    if len(s) < 3:
        return None
    vals = [p["value"] for p in s]
    plateau = st.median(sorted(vals)[-2:])
    if plateau <= 0:
        return None
    for p in s:
        if p["value"] >= frac * plateau:
            return {"knee_ctas": p["ctas"], "value_at_knee": p["value"],
                    "plateau": plateau, "frac": frac,
                    "monotone": vals == sorted(vals),
                    "max_ctas_in_series": s[-1]["ctas"]}
    return {"knee_ctas": None, "plateau": plateau, "frac": frac,
            "note": "no point reaches the plateau fraction", "monotone": False,
            "max_ctas_in_series": s[-1]["ctas"]}


def r2_linear_through_origin(xs, ys):
    """R^2 of y = a*x (one block per CU => throughput should be proportional)."""
    if len(xs) < 3 or not any(ys):
        return None
    a = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    ybar = st.mean(ys)
    ss_res = sum((y - a * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return None if ss_tot == 0 else 1.0 - ss_res / ss_tot


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    do_print = "--print" in sys.argv
    pts = load(src)
    if not pts:
        json.dump({"schema_version": SCHEMA, "points": [], "status": "empty"},
                  open(dst, "w"), indent=2)
        print("no points yet")
        return 0

    modes = sorted({p["mode"] for p in pts})
    derived = {"knees": {}, "concurrent_over_isolated": {},
               "compute_slowdown": {}, "protocol_over_payload": {},
               "hypotheses": {}}

    # ---------------- knees, one per (mode, mlp, fanout, concurrency, extras)
    seen = set()
    for p in pts:
        k = (p["mode"], p["mlp"], p["fanout"], p["concurrency"], p["protocol"],
             p["proto_g"], p["pushers"], p["row_bytes"], p["work_mode"])
        if k in seen:
            continue
        seen.add(k)
        s = series(pts, mode=k[0], mlp=k[1], fanout=k[2], concurrency=k[3],
                   protocol=k[4], proto_g=k[5], pushers=k[6], row_bytes=k[7],
                   work_mode=k[8])
        name = ("%s|mlp%d|%s|%s|p%d|g%d|n%d|rb%d|%s" % k)
        kn = knee(s)
        if kn:
            kn["n_points"] = len(s)
            derived["knees"][name] = kn

    # ---------------- concurrent / isolated, per pair_key
    by_pair = {}
    for p in pts:
        by_pair.setdefault(p["pair_key"], {})[p["concurrency"]] = p
    for pk, arms in sorted(by_pair.items()):
        iso, con, resv = arms.get("isolated"), arms.get("concurrent"), \
            arms.get("reserved_only")
        if iso and con and iso["value"] > 0:
            derived["concurrent_over_isolated"][pk] = {
                "ratio": round(con["value"] / iso["value"], 4),
                "isolated": iso["value"], "concurrent": con["value"],
                "metric": iso["metric"],
                "overlap_pct": con.get("overlap_pct_median"),
                "concurrent_is_really_concurrent":
                    (con.get("overlap_pct_median") or 0) >= 90.0,
            }
        # compute-role slowdown: the C-matched reserve-only twin is the control,
        # never the 256-CTA run (exp_20's lesson).
        if con and resv and (resv.get("cmp_tflops_median") or 0) > 0:
            derived["compute_slowdown"][pk] = {
                "live_over_control":
                    round(con["cmp_tflops_median"] / resv["cmp_tflops_median"], 4),
                "cmp_tflops_live": con["cmp_tflops_median"],
                "cmp_tflops_control": resv["cmp_tflops_median"],
                "cmp_tiles_matched": con["cmp_tiles"] == resv["cmp_tiles"],
            }

    # ---------------- Q2: what does a dedicated pool actually cost?
    # Two separable terms, both measured in the same launch family:
    #   reservation term  = compute lost by removing C CTAs from the pool
    #                       (reserved_only vs the full-grid shape-16 series)
    #   interference term = compute lost because those C CTAs push traffic
    #                       (concurrent vs its own C-matched reserved_only twin)
    # A dedicated pool is only defensible where the interference term is the
    # larger one; the reservation term is the part a fused design never pays.
    anchor = [p for p in pts if p["mode"] == "mfma" and p["mlp"] == 16
              and p["ctas"] == 256]
    full_rate = (anchor[0].get("cmp_tflops_median") or anchor[0]["value"]) \
        if anchor else None
    by_shape16_ctas = {p["ctas"]: (p.get("cmp_tflops_median") or p["value"])
                       for p in pts if p["mode"] == "mfma" and p["mlp"] == 16}
    q2 = derived["q2_dedicated_pool"] = {}
    for pk, arms in sorted(by_pair.items()):
        con, resv = arms.get("concurrent"), arms.get("reserved_only")
        if not (con and resv and full_rate):
            continue
        r_rate = resv.get("cmp_tflops_median") or 0.0
        x_rate = con.get("cmp_tflops_median") or 0.0
        if r_rate <= 0 or x_rate <= 0:
            continue
        c = resv["ctas"]
        grid = resv["grid"]
        reservation = 1.0 - r_rate / full_rate
        interference = (r_rate - x_rate) / full_rate
        twin = by_shape16_ctas.get(grid - c)
        q2[pk] = {
            "ctas_reserved": c,
            "grid": grid,
            "cmp_tflops_full_grid": round(full_rate, 2),
            "cmp_tflops_reserved_idle": round(r_rate, 2),
            "cmp_tflops_traffic_live": round(x_rate, 2),
            "reservation_cost_frac": round(reservation, 4),
            "interference_cost_frac": round(interference, 4),
            "total_dedicated_cost_frac": round(1.0 - x_rate / full_rate, 4),
            "cta_share_of_grid": round(c / float(grid), 4),
            # a same-shape compute-only launch at (grid - C) CTAs; if the idle
            # reservation is honest this should match reserved_only closely
            "compute_only_twin_tflops": None if twin is None else round(twin, 2),
            "twin_over_reserved": None if twin is None or r_rate <= 0
            else round(twin / r_rate, 4),
            "interference_exceeds_reservation": interference > reservation,
            "payload_delivered_GBps": con["value"],
        }
    if q2:
        wins = [k for k, v in q2.items() if v["interference_exceeds_reservation"]]
        derived["q2_summary"] = {
            "n_pairs": len(q2),
            "n_where_interference_exceeds_reservation": len(wins),
            "pairs_where_dedicating_could_pay": sorted(wins),
            "median_reservation_cost_frac":
                round(st.median([v["reservation_cost_frac"] for v in q2.values()]), 4),
            "median_interference_cost_frac":
                round(st.median([v["interference_cost_frac"] for v in q2.values()]), 4),
        }

    # ---------------- protocol vs payload at the same (ctas, mlp, fanout)
    for p in pts:
        if not p["protocol"]:
            continue
        base = [q for q in pts
                if q["protocol"] == 0 and q["mode"] == p["mode"]
                and q["ctas"] == p["ctas"] and q["mlp"] == p["mlp"]
                and q["fanout"] == p["fanout"]
                and q["concurrency"] == p["concurrency"]
                and q["row_bytes"] == p["row_bytes"]
                and q["pushers"] == p["pushers"]]
        if base and base[0]["value"] > 0:
            derived["protocol_over_payload"]["%s|c%d|%s|g%d|%s" % (
                p["mode"], p["ctas"], p["fanout"], p["proto_g"],
                p["concurrency"])] = {
                "ratio": round(p["value"] / base[0]["value"], 4),
                "payload_only": base[0]["value"], "with_protocol": p["value"],
            }

    # ---------------- H1-H4, adjudicated mechanically
    h = derived["hypotheses"]
    for mlp in (4, 8):
        s = series(pts, mode="xgmi", fanout="single", mlp=mlp,
                   concurrency="isolated", protocol=0, pushers=1,
                   row_bytes=14336, work_mode="per_cta")
        if not s:
            continue
        peak = 76.8
        reach = [p["ctas"] for p in s if p["value"] >= H1_FRAC * peak]
        h["H1_mlp%d" % mlp] = {
            "claim": "single-link push reaches %d%% of %.1f GB/s by %d CTAs"
                     % (H1_FRAC * 100, peak, H1_MAX_CTAS),
            "ctas_needed": min(reach) if reach else None,
            "verdict": ("SUPPORTED" if reach and min(reach) <= H1_MAX_CTAS
                        else "REFUTED" if reach else "REFUTED_NEVER_REACHED"),
            "best_value": max(p["value"] for p in s),
            "best_peak_fraction": max(p["peak_fraction"] for p in s),
            "falsifier_triggered": bool(
                (not reach) or min(reach) >= FALSIFIER_CTAS),
            "falsifier_meaning": ("if triggered: the pusher is bandwidth-limited, "
                                  "not protocol-limited, and A1's C=64 optimum "
                                  "inverts the M4-first priority"),
        }
    for mlp in (1, 4, 8):
        s = series(pts, mode="xgmi", fanout="rr7", mlp=mlp,
                   concurrency="isolated", protocol=0, pushers=1,
                   row_bytes=14336, work_mode="per_cta")
        if not s:
            continue
        reach = [p["ctas"] for p in s if p["value"] >= H1_FRAC * 537.6]
        h["H2_mlp%d" % mlp] = {
            "claim": "aggregate egress saturates by 16-32 CTAs",
            "ctas_needed_for_75pct_of_537.6": min(reach) if reach else None,
            "knee_90pct_of_plateau": (knee(s) or {}).get("knee_ctas"),
            "verdict": ("SUPPORTED" if reach and min(reach) <= H2_MAX_CTAS
                        else "REFUTED" if reach else "PLATEAU_BELOW_75PCT"),
            "best_value": max(p["value"] for p in s),
            "best_peak_fraction": max(p["peak_fraction"] for p in s),
        }
    ratios = [v["ratio"] for k, v in derived["concurrent_over_isolated"].items()
              if k.startswith("xgmi") and "|p0|" in k]
    proto_ratios = [v["ratio"] for v in derived["protocol_over_payload"].values()]
    if ratios:
        h["H3"] = {
            "claim": "the concurrent xGMI curve is depressed below isolated at "
                     "equal C, and the depression tracks protocol, not payload",
            "payload_only_concurrent_over_isolated_median": round(
                st.median(ratios), 4),
            "n": len(ratios),
            "depressed": st.median(ratios) < H3_DEPRESSION,
            "protocol_over_payload_median": (round(st.median(proto_ratios), 4)
                                             if proto_ratios else None),
            "verdict": ("SUPPORTED" if st.median(ratios) < H3_DEPRESSION
                        else "REFUTED_PAYLOAD_RIDES_FREE"),
            "reading": ("if payload-only concurrency is ~1.0 while the protocol "
                        "arm is well below 1.0, exp_20's protocol-not-payload "
                        "verdict reproduces as a curve, and a compute CTA can "
                        "carry payload for free"),
        }
    for shape in (4, 256):
        s = series(pts, mode="mfma", mlp=shape, concurrency="isolated")
        if len(s) < 3:
            continue
        r2 = r2_linear_through_origin([p["ctas"] for p in s],
                                      [p["value"] for p in s])
        h["H4_shape%d" % shape] = {
            "claim": "TFLOPS vs CTAs is linear through the origin (one block "
                     "per CU; no oversubscription regime on CDNA4)",
            "r2_through_origin": None if r2 is None else round(r2, 5),
            "verdict": ("SUPPORTED" if r2 is not None and r2 >= H4_R2
                        else "REFUTED"),
            "value_at_max_ctas": s[-1]["value"],
            "peak_fraction_at_max_ctas": s[-1]["peak_fraction"],
            "per_cta_efficiency_first_vs_last": round(
                (s[-1]["value"] / s[-1]["ctas"]) / (s[0]["value"] / s[0]["ctas"]),
                4) if s[0]["value"] else None,
        }

    doc = {
        "schema_version": SCHEMA,
        "experiment": "exp_22 resource saturation vs CTA count (paper Fig 2 / Q4)",
        "source_jsonl": src,
        "n_points": len(pts),
        "modes_present": modes,
        "status": "valid_diagnostic",
        "record_schema": {
            "mode": "mfma | hbm | xgmi",
            "ctas": "resource-role CTA count C (mode a: total CTAs)",
            "mlp": "mode c: store_peer_packets_multi depth; mode a: MFMAs per "
                   "16 KiB staged tile; mode b: 0",
            "fanout": "single (one xGMI link) | rr7 (aggregate egress) | na",
            "concurrency": "isolated | concurrent | reserved_only",
            "metric": "GBps | TFLOPS",
            "value": "headline number; wall-clock median for isolated, "
                     "role-span median for concurrent (value_source says which)",
            "repeats": "timed rotations behind value",
            "median": "median of values[]",
            "spread": "p25/p75/min/max/sd/rel_iqr_pct of values[]",
            "peak_fraction": "value / device peak for that resource class",
            "overlap_pct_median": "percent of the resource role's span during "
                                  "which the compute role was also resident -- "
                                  "the proof that a concurrent point is not "
                                  "secretly sequential",
            "cmp_tflops_median": "compute role's own throughput in the same "
                                 "launch; concurrent/reserved_only ratio is the "
                                 "compute slowdown",
            "res_rows / bytes / flops": "work actually done, for re-deriving "
                                        "every rate by hand",
        },
        "peaks_used": {
            "xgmi_link_GBps": 76.8, "xgmi_aggregate_egress_GBps": 537.6,
            "hbm_GBps": 8000.0, "bf16_dense_TFLOPS": 2300.0,
            "source": "plan.md reference ceilings; bf16 peak is the MI350X "
                      "spec sheet (dense, no sparsity). Override with "
                      "--peak-* if a measured ceiling replaces a spec one.",
        },
        "derived": derived,
        "points": pts,
    }
    with open(dst, "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote %s (%d points)" % (dst, len(pts)))

    if do_print:
        print("\n--- knees (smallest C reaching 90%% of plateau) ---")
        for k, v in sorted(derived["knees"].items()):
            print("  %-52s knee=%-5s plateau=%9.2f monotone=%s"
                  % (k, v.get("knee_ctas"), v.get("plateau", 0),
                     v.get("monotone")))
        if derived.get("q2_summary"):
            print("\n--- Q2: dedicated pool cost decomposition ---")
            for k, v in sorted(derived["q2_dedicated_pool"].items()):
                print("  %-46s C=%-4s reserve=%6.1f%% interfere=%6.1f%% "
                      "total=%6.1f%% twin/res=%s%s"
                      % (k, v["ctas_reserved"],
                         100 * v["reservation_cost_frac"],
                         100 * v["interference_cost_frac"],
                         100 * v["total_dedicated_cost_frac"],
                         v["twin_over_reserved"],
                         "  <<< interference dominates"
                         if v["interference_exceeds_reservation"] else ""))
            print("  summary: %s" % derived["q2_summary"])
        print("\n--- hypotheses ---")
        for k, v in sorted(derived["hypotheses"].items()):
            print("  %-14s %-24s %s" % (k, v.get("verdict"),
                                        {kk: vv for kk, vv in v.items()
                                         if kk not in ("claim", "reading",
                                                       "falsifier_meaning")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
