#!/usr/bin/env python3
"""Synthetic-data self-test for knees.py. CPU-only, no GPU, no node.

Exists because knees.py runs once, at the end of a 40-minute campaign, on data whose
shape it has never seen. A KeyError there costs the whole invocation. This builds a
saturation.json with the real schema and known-by-construction knees, runs knees.py
on it, and asserts the extracted knees are the planted ones.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CTAS_A = [8, 16, 32, 64, 96, 152, 200, 256, 304]
CTAS_B = [4, 8, 16, 32, 64, 128, 304]
CTAS_C = [1, 2, 4, 8, 16, 32, 64]


def pt(mode, overlay, ctas, depth, fanout, protocol, value, metric,
       gemm=None, slowdown=None, chk=True):
    return {
        "mode": mode, "ctas": ctas, "depth": depth, "fanout": fanout,
        "overlay": overlay, "protocol": protocol, "metric": metric,
        "payload_granularity": "fine", "value": value, "samples": [value],
        "per_rank": {"resource": [value] * 8, "gemm_tflops": [gemm] * 8 if gemm else [None] * 8},
        "concurrent_gemm_tflops": gemm, "gemm_tflops": gemm,
        "res_span_us": 1000.0, "gemm_span_us": 1000.0, "host_wall_ms": 2.0, "rounds": 1,
        "checksum_ok": chk if mode == "c" else None,
        "checksum_mismatches": 0 if mode == "c" else None,
        "checksum_fold": 0 if mode == "c" else None,
        **({"gemm_slowdown": slowdown} if slowdown else {}),
    }


def sat(c, knee, plateau):
    """Monotone saturating curve whose value first reaches 0.9*plateau exactly at C=knee."""
    return plateau * min(1.0, 0.9 * c / knee) if c <= knee else plateau * min(
        1.0, 0.9 + 0.1 * (c - knee) / max(1, 4 * knee))


def build(path):
    points = []
    for c in CTAS_A:                                    # perfectly linear MFMA
        points.append(pt("a", "isolated", c, None, None, False, 0.42 * c, "TFLOPS",
                         gemm=0.42 * c))
    for c in CTAS_B:                                    # HBM knee planted at 32
        points.append(pt("b", "isolated", c, None, None, False, sat(c, 32, 4900.0), "GBps"))
        if c < 304:
            points.append(pt("b", "concurrent", c, None, None, False,
                             0.82 * sat(c, 32, 4900.0), "GBps", gemm=90.0, slowdown=1.18))
            points.append(pt("b", "reserve_control", c, None, None, False, 106.0, "TFLOPS",
                             gemm=106.0))
    for fan in ("single", "rr7"):
        for d in (0, 1, 4, 8):
            base = 44.0 if fan == "single" else 210.0
            k = {0: 8, 1: 32, 4: 16, 8: 8}[d]           # depth=1 plants the falsifier at 32
            for proto in (False, True):
                for ov in ("isolated", "concurrent"):
                    for c in CTAS_C:
                        v = sat(c, k, base) * (0.88 if proto else 1.0) * (
                            0.79 if ov == "concurrent" else 1.0)
                        points.append(pt("c", ov, c, d, fan, proto, v, "GBps",
                                         gemm=(95.0 if ov == "concurrent" else None),
                                         slowdown=(1.09 if ov == "concurrent" else None)))
    for c in CTAS_C:
        points.append(pt("c", "reserve_control", c, 0, "single", False, 104.0, "TFLOPS",
                         gemm=104.0))
    json.dump({"experiment": "synthetic", "generated": "synthetic",
               "tick_rate_hz": 1.0e8, "ceilings": {}, "points": points},
              open(path, "w"), indent=1)


def main():
    tmp = tempfile.mkdtemp()
    inp, out = os.path.join(tmp, "saturation.json"), os.path.join(tmp, "knees.json")
    build(inp)
    rc = subprocess.call([sys.executable, os.path.join(HERE, "knees.py"),
                          "--inp", inp, "--out", out])
    if rc != 0:
        sys.exit(f"knees.py exited {rc}")
    k = json.load(open(out))
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    idx = {r["series"]: r for r in k["series"]}
    check(k["mode_a_linearity"]["max_abs_dev_pct"] < 0.01,
          f"linear input should show ~0% deviation, got {k['mode_a_linearity']['max_abs_dev_pct']}")
    check(idx["mode b isolated"]["knee_90"] == 32,
          f"planted mode-b knee 32, got {idx['mode b isolated']['knee_90']}")
    for d, want in ((0, 8), (1, 32), (4, 16), (8, 8)):
        s = f"mode c isolated depth={d} fanout=single proto=0"
        check(idx[s]["knee_90"] == want, f"{s}: planted knee {want}, got {idx[s]['knee_90']}")
    s = "mode c isolated depth=0 fanout=single proto=0"
    check(abs(idx[s]["conc_over_iso_at_knee"] - 0.79) < 1e-6,
          f"interference term should be 0.79, got {idx[s].get('conc_over_iso_at_knee')}")
    check(abs(idx[s]["protocol_on_over_off_at_knee"] - 0.88) < 1e-6,
          f"protocol gap should be 0.88, got {idx[s].get('protocol_on_over_off_at_knee')}")
    # depth=1 was planted at exactly 32, which is the falsifier threshold (>= 32)
    check("TRIGGERED" in k["falsifier_verdict"]["falsifier"],
          "a planted 32-CTA 75% crossing must trigger the falsifier")
    check(all(r["checksums_ok"] is not False for r in k["series"]),
          "no synthetic series should report a checksum failure")

    if fails:
        print("SELFTEST FAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"\nSELFTEST OK ({len(k['series'])} series parsed, all planted knees recovered)")


if __name__ == "__main__":
    main()
