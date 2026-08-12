#!/usr/bin/env python3
"""exp_22: events.json (any arm) -> timeline_bins.csv + integral validation.

This file owns the WHOLE traffic model, for both arms, so there is exactly one
place where a byte count is decided.  Every byte is analytic and computed on
the host from device-published counts (the resolved launch plan for arm (b),
the classified dispatch list for arm (a)).  There are no in-kernel byte
counters, by design: a counter is an atomic on the emit path, i.e. an
instrument that changes the quantity it measures.

Arm (b)'s per-phase bytes are EXACT, not averaged.  The kernel's tile->CTA map
is deterministic -- producer CTA `pid` owns tiles {pid + i*NG}, decoded by
`decode_tile(t, num_pid_m, cols, WGM)`; reducer CTA `pid_r` owns tiles
{pid_r + i*NR} -- so the host recomputes which tile each EMIT/REDUCE interval
belongs to and therefore its exact destination rank and valid column count.

  bin_timeline.py events_ours_s5.json events_b0_reference.json \
      --csv timeline_bins.csv --json validation.json
"""

import argparse
import csv
import json
import math
import os
import sys

BIN_US = 10.0
WORLD = 8
BF16 = 2

# --- phase ids, mirrored from hk_trace's enum ------------------------------
(CTA_BEG, MAINLOOP_BEG, MAINLOOP_END, CREDIT_WAIT_BEG, CREDIT_WAIT_END,
 EMIT_BEG, EMIT_END, RELEASE_BEG, RELEASE_END, PUBLISH_END,
 READY_WAIT_BEG, READY_WAIT_END, REDUCE_BEG, REDUCE_END,
 CREDIT_PUB_END, CTA_END) = range(16)


def decode_tile(t, num_pid_m, num_pid_n, wgm):
    """Python mirror of gemm_rs_mi300x.cpp's decode_tile. Must stay in sync."""
    in_group = wgm * num_pid_n
    group = t // in_group
    first = group * wgm
    gsize = min(num_pid_m - first, wgm)
    return first + (t % in_group) % gsize, (t % in_group) // gsize


# ---------------------------------------------------------------------------
# Arm (b): intervals with exact bytes
# ---------------------------------------------------------------------------
def ours_intervals(doc):
    plan, shape = doc["plan"], doc["shape"]
    m, n = shape["m"], shape["n"]
    k_local = int(plan["k_local"])
    bm, bn, bk = int(plan["bm"]), int(plan["bn"]), int(plan["bk"])
    ng = int(plan["num_gemm_ctas"])
    nr = int(plan["num_reducer_ctas"])
    eb, cols = int(plan["eb"]), int(plan["col_count"])
    num_pid_m = m // bm
    tiles = num_pid_m * cols
    wgm = 4 if tiles <= ng else num_pid_m
    ticks_per_us = doc["ticks_per_us"]
    if not ticks_per_us:
        raise SystemExit("events.json has no measured ticks_per_us; the x axis "
                         "would be unscaled. Re-run trace_run.py.")

    # Per-tile analytic quantities (design.md section 4).
    mainloop_flops = 2 * bm * bn * k_local
    mainloop_bytes = (bm + bn) * k_local * BF16 + (bm + bn) * bk * BF16

    rows = doc["ranks"]["0"]["events"]
    by_cta = {}
    for cta, phase, tick in rows:
        by_cta.setdefault(cta, []).append((phase, tick))

    out = []
    for cta, slots in by_cta.items():
        slots.sort(key=lambda s: s[1])
        producer = cta < ng
        if producer:
            owned = list(range(cta, tiles, ng))
        else:
            owned = list(range(cta - ng, int(plan["red_tiles"]), nr))
        open_at, seen_emit, seen_reduce, seen_mainloop = {}, 0, 0, 0
        for phase, tick in slots:
            us = tick / ticks_per_us
            if phase in (MAINLOOP_BEG, CREDIT_WAIT_BEG, EMIT_BEG, RELEASE_BEG,
                         READY_WAIT_BEG, REDUCE_BEG):
                open_at[phase] = us
                continue
            pairs = {MAINLOOP_END: MAINLOOP_BEG, CREDIT_WAIT_END: CREDIT_WAIT_BEG,
                     EMIT_END: EMIT_BEG, RELEASE_END: RELEASE_BEG,
                     READY_WAIT_END: READY_WAIT_BEG, REDUCE_END: REDUCE_BEG}
            if phase not in pairs or pairs[phase] not in open_at:
                continue
            begin = open_at.pop(pairs[phase])
            item = {"cta": cta, "begin_us": begin, "end_us": us,
                    "mfma": 0.0, "hbm_bytes": 0.0, "xgmi_bytes": 0.0,
                    "flops": 0.0}
            if phase == MAINLOOP_END:
                item["phase"] = "mainloop"
                item["mfma"] = 1.0
                item["hbm_bytes"] = mainloop_bytes
                item["flops"] = mainloop_flops
                seen_mainloop += 1
            elif phase == EMIT_END:
                item["phase"] = "emit"
                if seen_emit < len(owned):
                    tm, tn = decode_tile(owned[seen_emit], num_pid_m, cols, wgm)
                    valid = min(bn, n - tn * bn)
                    payload = bm * valid * BF16
                    dest = (tm * bm) // (m // WORLD)
                    me = 0                      # rank 0 is the plotted rank
                    if dest == me:
                        item["hbm_bytes"] = payload
                    else:
                        item["xgmi_bytes"] = payload
                seen_emit += 1
            elif phase == REDUCE_END:
                item["phase"] = "reduce"
                if seen_reduce < len(owned):
                    col = owned[seen_reduce] % cols
                    valid = min(bn, n - col * bn)
                    item["hbm_bytes"] = (WORLD + 1) * eb * valid * BF16
                seen_reduce += 1
            elif phase == CREDIT_WAIT_END:
                item["phase"] = "credit_wait"
            elif phase == READY_WAIT_END:
                item["phase"] = "ready_wait"
            elif phase == RELEASE_END:
                item["phase"] = "release"
            out.append(item)

    meta = {
        "mfma_denominator": ng,
        "mfma_denominator_note": "304 - NR: the CTAs that run a mainloop at "
                                 "all. Dividing by 304 would cap the proxy "
                                 "below 1 and invent an idle band.",
        "expected_flops": 2 * m * n * k_local,
        "expected_hbm_bytes": (
            tiles * mainloop_bytes                       # mainloop loads
            + m * n * BF16 // WORLD                      # emit, local eighth
            + (WORLD + 1) * (m // WORLD) * n * BF16),    # reduce read + write
        "expected_xgmi_bytes": (WORLD - 1) * m * n * BF16 // WORLD,
        "wgm": wgm, "tiles": tiles, "num_pid_m": num_pid_m,
    }
    return out, meta


# ---------------------------------------------------------------------------
# Arm (a): classified dispatch intervals, class totals spread by duration
# ---------------------------------------------------------------------------
def b0_intervals(doc):
    shape = doc["shape"]
    m, n, k = shape["m"], shape["n"], shape["k"]
    k_local = k // WORLD
    out_bytes = m * n * BF16                    # the materialized partial

    # Analytic class totals for the reference implementation.  Named terms so
    # result.md can quote the model rather than a single opaque number.
    #   mfma  : one bf16 GEMM producing an M x N partial into HBM
    #   hbm   : the bias add / any elementwise pass (read + write the partial)
    #   xgmi  : ring reduce-scatter moves (world-1)/world of the buffer off
    #           rank -- exactly the 58.72 MB our epilogue moves on shape 5,
    #           which is what makes the two columns comparable at all
    totals = {
        "mfma": {
            "flops": 2 * m * n * k_local,
            "hbm_bytes": (m + n) * k_local * BF16 + out_bytes,
            "xgmi_bytes": 0,
        },
        "hbm": {
            "flops": 0,
            "hbm_bytes": 2 * out_bytes if shape.get("has_bias") else 0,
            "xgmi_bytes": 0,
        },
        "xgmi": {
            "flops": 0,
            "hbm_bytes": 2 * (WORLD - 1) * out_bytes // WORLD + out_bytes // WORLD,
            "xgmi_bytes": (WORLD - 1) * out_bytes // WORLD,
        },
        "unclassified": {"flops": 0, "hbm_bytes": 0, "xgmi_bytes": 0},
    }

    per_class_ns = {}
    for item in doc["intervals"]:
        per_class_ns.setdefault(item["resource"], 0)
        per_class_ns[item["resource"]] += item["end_ns"] - item["begin_ns"]

    out = []
    for item in doc["intervals"]:
        klass = item["resource"]
        duration = item["end_ns"] - item["begin_ns"]
        share = duration / per_class_ns[klass] if per_class_ns.get(klass) else 0.0
        totals_k = totals.get(klass, totals["unclassified"])
        out.append({
            "cta": 0,
            "phase": klass,
            "name": item["name"],
            "begin_us": item["begin_ns"] / 1e3,
            "end_us": item["end_ns"] / 1e3,
            "mfma": 1.0 if klass == "mfma" else 0.0,
            "hbm_bytes": totals_k["hbm_bytes"] * share,
            "xgmi_bytes": totals_k["xgmi_bytes"] * share,
            "flops": totals_k["flops"] * share,
        })
    meta = {
        "mfma_denominator": 1,
        "mfma_denominator_note": "arm (a) strips are binary occupancy bars: a "
                                 "kernel of a class either owns the device or "
                                 "does not. That non-overlap IS the finding.",
        "expected_flops": totals["mfma"]["flops"],
        "expected_hbm_bytes": sum(t["hbm_bytes"] for t in totals.values()),
        "expected_xgmi_bytes": sum(t["xgmi_bytes"] for t in totals.values()),
        "class_totals": totals,
    }
    return out, meta


# ---------------------------------------------------------------------------
def bin_arm(intervals, meta, bin_us=BIN_US):
    lo = min(i["begin_us"] for i in intervals)
    hi = max(i["end_us"] for i in intervals)
    count = max(1, int(math.ceil((hi - lo) / bin_us)))
    mfma = [0.0] * count
    hbm = [0.0] * count
    xgmi = [0.0] * count
    flops = [0.0] * count
    for item in intervals:
        begin, end = item["begin_us"] - lo, item["end_us"] - lo
        duration = max(end - begin, 1e-9)
        first, last = int(begin // bin_us), int(min(end, hi - lo) // bin_us)
        for b in range(first, min(last + 1, count)):
            b0, b1 = b * bin_us, (b + 1) * bin_us
            overlap = max(0.0, min(end, b1) - max(begin, b0))
            if overlap <= 0:
                continue
            mfma[b] += item["mfma"] * overlap / bin_us
            hbm[b] += item["hbm_bytes"] * (overlap / duration) / (bin_us * 1e-6) / 1e9
            xgmi[b] += item["xgmi_bytes"] * (overlap / duration) / (bin_us * 1e-6) / 1e9
            flops[b] += item["flops"] * (overlap / duration)
    denom = meta["mfma_denominator"]
    return {
        "t0_us": lo, "bin_us": bin_us, "bins": count,
        "mfma": [v / denom for v in mfma],
        "hbm_gbs": hbm, "xgmi_gbs": xgmi, "flops": flops,
    }


def validate(binned, meta, tolerance=0.10):
    seconds = binned["bin_us"] * 1e-6
    got = {
        "hbm_bytes": sum(binned["hbm_gbs"]) * 1e9 * seconds,
        "xgmi_bytes": sum(binned["xgmi_gbs"]) * 1e9 * seconds,
        "flops": sum(binned["flops"]),
    }
    report = {}
    for key, expected in (("hbm_bytes", meta["expected_hbm_bytes"]),
                          ("xgmi_bytes", meta["expected_xgmi_bytes"]),
                          ("flops", meta["expected_flops"])):
        error = (got[key] - expected) / expected if expected else 0.0
        report[key] = {"expected": expected, "integral": got[key],
                       "rel_error": error, "pass": abs(error) <= tolerance}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--csv", default="timeline_bins.csv")
    ap.add_argument("--json", default="validation.json")
    ap.add_argument("--bin-us", type=float, default=BIN_US)
    args = ap.parse_args()

    rows, validation = [], {}
    for path in args.events:
        with open(path) as handle:
            doc = json.load(handle)
        arm = doc["arm"]
        if doc["kind"] == "phase_ring":
            intervals, meta = ours_intervals(doc)
        else:
            intervals, meta = b0_intervals(doc)
        binned = bin_arm(intervals, meta, args.bin_us)
        report = validate(binned, meta)
        validation[arm] = {
            "source": os.path.basename(path),
            "meta": {k: v for k, v in meta.items() if k != "class_totals"},
            "class_totals": meta.get("class_totals"),
            "integrals": report,
            "epoch_us": binned["bins"] * binned["bin_us"],
            "all_pass": all(v["pass"] for v in report.values()),
        }
        for b in range(binned["bins"]):
            rows.append({
                "arm": arm,
                "bin_index": b,
                "t_us_start": round(b * binned["bin_us"], 3),
                "t_us_end": round((b + 1) * binned["bin_us"], 3),
                "mfma_frac": round(binned["mfma"][b], 6),
                "mfma_denominator": meta["mfma_denominator"],
                "hbm_gbs": round(binned["hbm_gbs"][b], 3),
                "xgmi_gbs": round(binned["xgmi_gbs"][b], 3),
                "tflops": round(binned["flops"][b] / (binned["bin_us"] * 1e-6) / 1e12, 3),
            })
        print(f"{arm:<14} {binned['bins']} bins x {binned['bin_us']} us "
              f"= {binned['bins'] * binned['bin_us']:.0f} us")
        for key, value in report.items():
            flag = "PASS" if value["pass"] else "FAIL"
            print(f"  integral {key:<11} {flag} rel_err={value['rel_error']:+.2%}")

    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.json, "w") as handle:
        json.dump({"schema": "exp22.validation.v1",
                   "bin_us": args.bin_us,
                   "integral_tolerance": 0.10,
                   "arms": validation}, handle, indent=2)
    print(f"wrote {args.csv} ({len(rows)} rows) and {args.json}")
    return 0 if all(v["all_pass"] for v in validation.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
