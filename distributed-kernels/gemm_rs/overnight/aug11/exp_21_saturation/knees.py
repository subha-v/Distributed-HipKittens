#!/usr/bin/env python3
"""Knee extraction for exp_21. CPU-only: reads saturation.json, writes knees.json.

A "series" is one curve in the figure: everything except the CTA count held fixed
(mode, overlay, protocol, depth, fanout). For each series we report

  plateau   max value over C. Not a fitted asymptote -- the sweep's top C is a real
            measured point, and a fit would smuggle a model into a figure whose
            whole purpose is to show where the measurement stops rising.
  knee_90   smallest C whose value >= 0.90 * plateau. This is the headline number:
            the paper's "topology-correct pool size".
  knee_75   smallest C reaching 0.75 * plateau. Exists only to adjudicate the
            pre-registered falsifier (plan.md): if the ISOLATED single-fanout curve
            needs C >= 32 to reach 75% of its own plateau at ANY depth, the
            tiny-pool claim is wrong for our emit shape.

Derived comparisons at a series' knee:
  conc_over_iso   concurrent / isolated at the same C, depth, fanout, protocol.
                  The interference term.
  protocol_gap    protocol=1 / protocol=0 at the same C, depth, fanout, overlay.
  gemm_slowdown   C-matched reserve control TFLOPS / concurrent-with-traffic TFLOPS
                  on the SAME 304-C GEMM CTAs. Never against the full 304 grid.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def key(p):
    return (p["mode"], p["overlay"], bool(p["protocol"]), p["depth"], p["fanout"])


def label(k):
    mode, overlay, proto, depth, fan = k
    bits = [f"mode {mode}", overlay]
    if mode == "c":
        bits += [f"depth={depth}", f"fanout={fan}", f"proto={int(proto)}"]
    return " ".join(bits)


def series_of(points):
    out = {}
    for p in points:
        out.setdefault(key(p), []).append(p)
    for k in out:
        out[k].sort(key=lambda p: p["ctas"])
    return out


def knee(curve, frac):
    """Smallest C at or above frac * plateau, plateau = max measured value."""
    vals = [(p["ctas"], p["value"]) for p in curve if p["value"] > 0]
    if not vals:
        return None, None, None
    plateau = max(v for _, v in vals)
    target = frac * plateau
    hit = next((c for c, v in vals if v >= target), None)
    return hit, plateau, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(HERE, "saturation.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "knees.json"))
    args = ap.parse_args()

    with open(args.inp) as fh:
        data = json.load(fh)
    points = data["points"]
    ser = series_of(points)
    by = {(p["mode"], p["overlay"], bool(p["protocol"]), p["depth"], p["fanout"],
           p["ctas"]): p for p in points}

    rows = []
    for k, curve in sorted(ser.items(), key=lambda kv: label(kv[0])):
        k90, plateau, _ = knee(curve, 0.90)
        k75, _, _ = knee(curve, 0.75)
        row = {
            "series": label(k), "mode": k[0], "overlay": k[1], "protocol": k[2],
            "depth": k[3], "fanout": k[4],
            "metric": curve[0]["metric"],
            "ctas": [p["ctas"] for p in curve],
            "values": [round(p["value"], 3) for p in curve],
            "plateau": round(plateau, 3) if plateau else None,
            "knee_90": k90, "knee_75": k75,
            "checksums_ok": (None if all(p["checksum_ok"] is None for p in curve)
                             else all(p["checksum_ok"] is not False for p in curve)),
        }
        if k90 is not None:
            at = by.get((k[0], k[1], k[2], k[3], k[4], k90))
            row["value_at_knee"] = round(at["value"], 3)
            row["res_span_us_at_knee"] = round(at["res_span_us"], 2)
            row["gemm_slowdown_at_knee"] = at.get("gemm_slowdown")
            # interference term: concurrent vs isolated at the SAME C
            other = by.get((k[0], "concurrent" if k[1] == "isolated" else "isolated",
                            k[2], k[3], k[4], k90))
            if other and other["value"] > 0 and k[1] == "isolated":
                row["conc_over_iso_at_knee"] = round(other["value"] / at["value"], 4)
            # protocol gap at the same C
            pg = by.get((k[0], k[1], not k[2], k[3], k[4], k90))
            if pg and pg["value"] > 0 and k[2] is False:
                row["protocol_on_over_off_at_knee"] = round(pg["value"] / at["value"], 4)
        rows.append(row)

    # ---- mode a linearity. Structural, not empirical: mode a is 1 CTA/CU from its
    # register tuple AND from its 64 KB LDS request, so nothing else can occupy a CU.
    # The measurement confirms the structure; it does not discover scaling.
    a = ser.get(("a", "isolated", False, None, None), [])
    lin = None
    if len(a) >= 2:
        per_cta = [(p["ctas"], p["value"] / p["ctas"]) for p in a if p["value"] > 0]
        ref = per_cta[-1][1]
        lin = {
            "per_cta_tflops": [[c, round(v, 5)] for c, v in per_cta],
            "ref_per_cta_at_max_c": round(ref, 5),
            "max_abs_dev_pct": round(max(abs(v - ref) / ref for _, v in per_cta) * 100, 2),
            "knee_90": knee(a, 0.90)[0],
            "note": ("1 CTA/CU is forced by the register tuple and by the 65,536 B dynamic "
                     "LDS request, so linear scaling to 304 is structural. Deviation is "
                     "measured against the largest-C per-CTA rate."),
        }

    # ---- falsifier: isolated mode-c single-fanout, any depth, needs C >= 32 for 75%
    fals = []
    for k, curve in ser.items():
        if k[0] == "c" and k[1] == "isolated" and k[4] == "single":
            k75, _, _ = knee(curve, 0.75)
            fals.append({"series": label(k), "knee_75": k75,
                         "triggers": bool(k75 is not None and k75 >= 32)})
    verdict = {
        "falsifier": ("TRIGGERED -- the tiny-pool claim fails for this emit shape and the "
                      "in-flight-bound priority inverts"
                      if any(f["triggers"] for f in fals) else
                      "NOT triggered -- every isolated single-link curve reaches 75% of its "
                      "own plateau below 32 CTAs"),
        "detail": fals,
    }

    out = {
        "experiment": "exp_21_saturation",
        "source": os.path.basename(args.inp),
        "generated_from": data.get("generated"),
        "payload_granularity": points[0]["payload_granularity"] if points else None,
        "tick_rate_hz": data.get("tick_rate_hz"),
        "ceilings": data.get("ceilings"),
        "definitions": {
            "plateau": "max measured value over C (no fitted asymptote)",
            "knee_90": "smallest C with value >= 0.90 * plateau -- the headline pool size",
            "knee_75": "smallest C with value >= 0.75 * plateau -- falsifier adjudication only",
            "conc_over_iso_at_knee": "concurrent / isolated at the isolated series' knee C",
            "gemm_slowdown_at_knee": "C-matched reserve-control TFLOPS / concurrent TFLOPS",
        },
        "mode_a_linearity": lin,
        "falsifier_verdict": verdict,
        "series": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    # One tidy row per measured point, for whatever plots the paper draws. Long format
    # on purpose: every panel in Fig 2 is a filter over these columns.
    csv_path = os.path.splitext(args.out)[0].replace("knees", "saturation") + ".csv"
    cols = ["mode", "overlay", "protocol", "depth", "fanout", "ctas", "metric", "value",
            "res_span_us", "gemm_span_us", "concurrent_gemm_tflops", "gemm_slowdown",
            "rounds", "checksum_ok", "payload_granularity"]
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for p in sorted(points, key=lambda p: (p["mode"], p["fanout"] or "", p["depth"] or 0,
                                               p["overlay"], p["protocol"], p["ctas"])):
            fh.write(",".join("" if p.get(c) is None else str(p.get(c)) for c in cols) + "\n")
    print(f"-> {csv_path} ({len(points)} rows)")

    print(f"{'series':<52} {'metric':<7} {'plateau':>9} {'knee90':>7} {'knee75':>7} "
          f"{'@knee':>9} {'c/i':>7} {'proto':>7}")
    for r in rows:
        print(f"{r['series']:<52} {r['metric']:<7} "
              f"{(r['plateau'] if r['plateau'] else 0):9.2f} "
              f"{str(r['knee_90']):>7} {str(r['knee_75']):>7} "
              f"{r.get('value_at_knee', 0):9.2f} "
              f"{r.get('conc_over_iso_at_knee', '-')!s:>7} "
              f"{r.get('protocol_on_over_off_at_knee', '-')!s:>7}")
    if lin:
        print(f"\nmode a: per-CTA TFLOPS deviates at most {lin['max_abs_dev_pct']}% from the "
              f"C={a[-1]['ctas']} rate; 90%-of-plateau knee at C={lin['knee_90']}")
    print(f"\nfalsifier: {verdict['falsifier']}")
    bad = [r["series"] for r in rows if r["checksums_ok"] is False]
    if bad:
        print(f"\n!! CHECKSUM FAILURES in: {bad}", file=sys.stderr)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
