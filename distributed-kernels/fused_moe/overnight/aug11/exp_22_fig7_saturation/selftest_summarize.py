#!/usr/bin/env python3
"""CPU self-test for summarize.py: synthesize two worlds and check that the
H1-H4 adjudication actually flips.

World A ("saturates early"): the single-link push hits 70 GB/s at 4 CTAs,
aggregate egress plateaus by 16, payload-only concurrency is free, protocol
costs 40%, MFMA is linear. Expect H1/H2 SUPPORTED, H3 REFUTED_PAYLOAD_RIDES_FREE
(payload rides free -- which is the paper's claim), H4 SUPPORTED.

World B ("bandwidth-limited"): the push climbs linearly to 64 CTAs and never
plateaus, payload-only concurrency is halved, MFMA saturates at 96 CTAs. Expect
H1 REFUTED with the falsifier triggered, H2 PLATEAU_BELOW_75PCT, H3 SUPPORTED,
H4 REFUTED.

If a hypothesis cannot be made to fail by any input, the measurement cannot
falsify it and the experiment is worthless. This is that check.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def rec(mode, ctas, mlp, fanout, conc, value, metric="GBps", protocol=0,
        proto_g=0, overlap=98.0, cmp_tf=0.0, peak=76.8, pushers=1, rb=14336):
    vals = [value * (1.0 + 0.004 * (i - 2)) for i in range(5)]
    key = "%s|c%d|m%d|%s|%s|p%d|g%d|n%d|rb%d|per_cta|u1" % (
        mode, ctas, mlp, fanout, conc, protocol, proto_g, pushers, rb)
    pair = "%s|c%d|m%d|%s|p%d|g%d|n%d|rb%d|per_cta|u1" % (
        mode, ctas, mlp, fanout, protocol, proto_g, pushers, rb)
    return {
        "key": key, "pair_key": pair, "mode": mode, "ctas": ctas, "mlp": mlp,
        "fanout": fanout, "concurrency": conc, "metric": metric,
        "value": value, "value_source": "wall", "repeats": 5,
        "median": value,
        "spread": {"p25": min(vals), "p75": max(vals), "min": min(vals),
                   "max": max(vals), "sd": 0.01 * value, "rel_iqr_pct": 0.8},
        "peak_fraction": value / peak, "values": vals,
        "overlap_pct_median": overlap, "cmp_tflops_median": cmp_tf,
        "cmp_tiles": 1000, "protocol": protocol, "proto_g": proto_g,
        "pushers": pushers, "row_bytes": rb, "work_mode": "per_cta",
        "wall_us_median": 30000.0, "res_rows": 100000, "bytes": 1e9,
        "flops": 0.0, "grid": 256,
    }


def world_a():
    pts = []
    for c, v in ((1, 20.0), (2, 38.0), (4, 60.0), (8, 71.0), (16, 73.0),
                 (32, 73.5), (64, 73.6)):
        for mlp in (4, 8):
            pts.append(rec("xgmi", c, mlp, "single", "isolated", v))
            pts.append(rec("xgmi", c, mlp, "single", "concurrent", v * 0.99,
                           cmp_tf=900.0))
            pts.append(rec("xgmi", c, mlp, "single", "reserved_only", 0.0,
                           cmp_tf=910.0))
            pts.append(rec("xgmi", c, mlp, "single", "isolated", v * 0.6,
                           protocol=1, proto_g=16))
    for c, v in ((1, 60.0), (2, 120.0), (4, 240.0), (8, 400.0), (16, 470.0),
                 (32, 480.0), (64, 482.0)):
        for mlp in (1, 4, 8):
            pts.append(rec("xgmi", c, mlp, "rr7", "isolated", v, peak=537.6))
    for c in (32, 64, 96, 128, 160, 192, 224, 256):
        for shape in (4, 256):
            pts.append(rec("mfma", c, shape, "na", "isolated", 7.0 * c,
                           metric="TFLOPS", peak=2300.0))
    return pts


def world_b():
    pts = []
    for c, v in ((1, 0.9), (2, 1.8), (4, 3.6), (8, 7.2), (16, 14.0),
                 (32, 27.0), (64, 52.0)):
        for mlp in (4, 8):
            pts.append(rec("xgmi", c, mlp, "single", "isolated", v))
            pts.append(rec("xgmi", c, mlp, "single", "concurrent", v * 0.5,
                           cmp_tf=700.0))
            pts.append(rec("xgmi", c, mlp, "single", "reserved_only", 0.0,
                           cmp_tf=1000.0))
    for c, v in ((1, 6.0), (2, 12.0), (4, 24.0), (8, 48.0), (16, 90.0),
                 (32, 170.0), (64, 300.0)):
        for mlp in (1, 4, 8):
            pts.append(rec("xgmi", c, mlp, "rr7", "isolated", v, peak=537.6))
    for c in (32, 64, 96, 128, 160, 192, 224, 256):
        for shape in (4, 256):
            v = 7.0 * min(c, 96)
            pts.append(rec("mfma", c, shape, "na", "isolated", v,
                           metric="TFLOPS", peak=2300.0))
    return pts


def run(pts, tag):
    d = tempfile.mkdtemp()
    jl = os.path.join(d, "in.jsonl")
    js = os.path.join(d, "out.json")
    with open(jl, "w") as fh:
        for p in pts:
            fh.write(json.dumps(p) + "\n")
    subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), jl, js],
                   check=True, stdout=subprocess.DEVNULL)
    doc = json.load(open(js))
    h = doc["derived"]["hypotheses"]
    print("  %s:" % tag)
    for k in sorted(h):
        print("    %-14s %s" % (k, h[k]["verdict"]))
    return h, doc


def main():
    ha, da = run(world_a(), "world A (saturates early, payload rides free)")
    hb, db = run(world_b(), "world B (bandwidth-limited, interference)")

    checks = [
        ("H1 flips", ha["H1_mlp4"]["verdict"] == "SUPPORTED"
         and hb["H1_mlp4"]["verdict"] != "SUPPORTED"),
        ("H1 falsifier fires only in B",
         not ha["H1_mlp4"]["falsifier_triggered"]
         and hb["H1_mlp4"]["falsifier_triggered"]),
        ("H2 flips", ha["H2_mlp4"]["verdict"] == "SUPPORTED"
         and hb["H2_mlp4"]["verdict"] != "SUPPORTED"),
        ("H3 flips", ha["H3"]["verdict"].startswith("REFUTED")
         and hb["H3"]["verdict"] == "SUPPORTED"),
        ("H4 flips", ha["H4_shape4"]["verdict"] == "SUPPORTED"
         and hb["H4_shape4"]["verdict"] == "REFUTED"),
        ("knees found in A",
         len(da["derived"]["knees"]) > 0),
        ("compute slowdown computed",
         len(da["derived"]["compute_slowdown"]) > 0),
        ("protocol ratio computed",
         len(da["derived"]["protocol_over_payload"]) > 0),
        ("overlap flag honoured",
         all(v["concurrent_is_really_concurrent"]
             for v in da["derived"]["concurrent_over_isolated"].values())),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", n))
    print("SELFTEST_SUMMARIZE:", "FAIL " + "; ".join(bad) if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
