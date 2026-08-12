#!/usr/bin/env python3
"""exp_22: per-phase profile from the ring, and the exp_20 confrontation.

The figure's load-bearing claim is that our kernel overlaps what the reference
serializes. That claim is only worth anything if the phases the ring draws agree
with an instrument built on a completely different principle -- exp_20's
macro-gated ablation, which measures a pool by DELETING it from the kernel and
timing the difference. This script puts the two side by side.

The comparable statistic needs care. exp_20's pools are whole-kernel deltas in
microseconds. The ring measures per-CTA intervals, and all producers run
concurrently, so the kernel-time contribution of a phase is not the sum over
CTAs -- it is roughly what a typical CTA spends, which is why the MEAN per-CTA
sum is quoted against exp_20 and the max is quoted beside it as the tail.

  phase_summary.py events_ours_s5.json [more...] [--json phase_summary.json]
"""

import argparse
import json
import statistics
import sys

PHASES = [
    "CTA_BEG", "MAINLOOP_BEG", "MAINLOOP_END", "CREDIT_WAIT_BEG",
    "CREDIT_WAIT_END", "EMIT_BEG", "EMIT_END", "RELEASE_BEG", "RELEASE_END",
    "PUBLISH_END", "READY_WAIT_BEG", "READY_WAIT_END", "REDUCE_BEG",
    "REDUCE_END", "CREDIT_PUB_END", "CTA_END",
]
IDX = {name: i for i, name in enumerate(PHASES)}

# (label, begin_id, end_id). Every pair a CTA can open and close.
PAIRS = [
    ("mainloop",    "MAINLOOP_BEG",    "MAINLOOP_END"),
    ("credit_wait", "CREDIT_WAIT_BEG", "CREDIT_WAIT_END"),
    ("emit",        "EMIT_BEG",        "EMIT_END"),
    ("release",     "RELEASE_BEG",     "RELEASE_END"),
    ("ready_wait",  "READY_WAIT_BEG",  "READY_WAIT_END"),
    ("reduce",      "REDUCE_BEG",      "REDUCE_END"),
]

# exp_20's macro-gated attribution, us. The `sync` pool is the one that the
# CREDIT_WAIT pair was added to make visible.
EXP20_SYNC_US = {"s5": 52.9, "s6": 168.6}
EXP20_RELEASE_US = {"s5": 65.4}


def pair_durations(events, ticks_per_us):
    """cta -> {label: (total_us, count)} by walking each CTA's ordered slots."""
    per_cta = {}
    for cta, pid, tick in events:
        per_cta.setdefault(cta, []).append((pid, tick))
    out = {}
    for cta, slots in per_cta.items():
        slots.sort(key=lambda s: s[1])
        totals = {}
        for label, begin, end in PAIRS:
            b_id, e_id = IDX[begin], IDX[end]
            open_tick, total, count = None, 0, 0
            for pid, tick in slots:
                if pid == b_id:
                    open_tick = tick
                elif pid == e_id and open_tick is not None:
                    total += tick - open_tick
                    open_tick = None
                    count += 1
            if count:
                totals[label] = (total / ticks_per_us, count)
        out[cta] = totals
    return out


def summarize(path):
    doc = json.load(open(path))
    tag = "s%d" % doc.get("plan", {}).get("config_row", 0)
    ticks_per_us = doc["ticks_per_us"]
    events = doc["ranks"]["0"]["events"]
    per_cta = pair_durations(events, ticks_per_us)
    nr = doc["plan"]["num_reducer_ctas"]
    n_gemm = doc["plan"]["num_gemm_ctas"]

    shape = doc["shape"]
    print(f"\n===== {path.split('/')[-1]}  "
          f"{shape['m']}x{shape['n']}x{shape['k']}  row {doc['plan']['config_row']}  "
          f"NR={nr}  producers={n_gemm} =====")
    print(f"  device_us rank0 {doc['device_us']['rank0']:.1f}, "
          f"ticks/us {ticks_per_us:.4f}, CTAs seen {len(per_cta)}, "
          f"drops {doc['ranks']['0']['drops']}")

    rows = {}
    header = (f"  {'phase':<12}{'CTAs':>6}{'events':>8}{'mean us':>10}"
              f"{'median':>9}{'max':>9}{'ev/CTA':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, _, _ in PAIRS:
        values = [t[label][0] for t in per_cta.values() if label in t]
        counts = [t[label][1] for t in per_cta.values() if label in t]
        if not values:
            continue
        rows[label] = {
            "ctas": len(values),
            "events": sum(counts),
            "mean_us": statistics.fmean(values),
            "median_us": statistics.median(values),
            "max_us": max(values),
            "events_per_cta": sum(counts) / len(counts),
        }
        r = rows[label]
        print(f"  {label:<12}{r['ctas']:>6}{r['events']:>8}{r['mean_us']:>10.1f}"
              f"{r['median_us']:>9.1f}{r['max_us']:>9.1f}"
              f"{r['events_per_cta']:>8.1f}")

    verdicts = []
    if tag in EXP20_SYNC_US and "credit_wait" in rows:
        measured = rows["credit_wait"]["mean_us"]
        expected = EXP20_SYNC_US[tag]
        ratio = measured / expected
        verdict = "AGREE" if 0.5 <= ratio <= 2.0 else "DISAGREE"
        verdicts.append({"quantity": "credit_wait vs exp_20 sync",
                         "measured_us": measured, "exp20_us": expected,
                         "ratio": ratio, "verdict": verdict})
        print(f"  credit_wait mean {measured:.1f} us vs exp_20 sync pool "
              f"{expected} us -> {ratio:.2f}x  {verdict}")
    if tag in EXP20_RELEASE_US and "release" in rows:
        measured = rows["release"]["mean_us"]
        print(f"  release mean {measured:.1f} us vs exp_20's pre-exp_26 release "
              f"pool {EXP20_RELEASE_US[tag]} us; exp_26 halved the release count "
              f"on this shape (rgroup 1 -> 2, -6.56%), so a LOWER number here is "
              f"the shipped behaviour, not a discrepancy")

    return {"file": path, "tag": tag, "shape": shape,
            "num_reducer_ctas": nr, "num_gemm_ctas": n_gemm,
            "mfma_denominator": n_gemm,
            "ticks_per_us": ticks_per_us,
            "device_us_rank0": doc["device_us"]["rank0"],
            "correctness": doc["correctness"],
            "drops": doc["ranks"]["0"]["drops"],
            "provenance": doc.get("provenance"),
            "phases": rows, "verdicts": verdicts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--json", default="phase_summary.json")
    args = ap.parse_args()
    out = [summarize(p) for p in args.events]
    with open(args.json, "w") as handle:
        json.dump({"schema": "exp22.phase_summary.v1",
                   "note": "mean is the per-CTA sum averaged over CTAs that ran "
                           "the phase; producers run concurrently so the mean is "
                           "the quantity comparable to exp_20's whole-kernel "
                           "pool, and max is the tail beside it",
                   "arms": out}, handle, indent=2)
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
