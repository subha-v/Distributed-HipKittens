#!/usr/bin/env python3
"""exp_22 arm (a): rocprofv3 kernel trace -> resource-class map + events.json.

Two jobs:

  1. Derive the kernel-name -> resource-class map from THE NAMES THAT ACTUALLY
     APPEAR in the captured trace.  Any name matching no rule is recorded under
     `unclassified` with its total time and dispatch count, never silently
     bucketed -- a mutually-exclusive-strips claim built on a guessed mapping
     is not evidence.
  2. Segment the trace into epochs and emit one epoch's intervals in the same
     envelope arm (b) uses, so bin_timeline.py can treat both identically.

Epoch segmentation needs no host/device clock correlation: b0_ref.py brackets
every measured epoch with a full device sync, so epochs are separated by gaps
far larger than any intra-epoch gap.  The host boundaries are read only as a
cross-check on the epoch count and duration.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

# Rules in priority order, applied to the lowercased kernel name.  Seeded from
# the sibling's rccl*/mori* -> xGMI, *gemm*/*mfma* -> MFMA, scatter/quant/
# combine/copy -> HBM, then CORRECTED against the eight names this node's
# reference stack actually emits (see result.md for the observed inventory):
#
#   ncclDevKernel_Generic_2   the reduce-scatter.  The seed rules looked for
#                             `rccl*` and `ncclkernel`; ROCm's RCCL exports the
#                             upstream NCCL symbol names, so neither matched and
#                             the single most important kernel in the arm landed
#                             in `unclassified`.  That is the rule working: it
#                             surfaced the gap instead of bucketing it.
#   __amd_rocclr_copyBuffer   866 dispatches, and fillBufferAligned 552 more --
#   __amd_rocclr_*            HIP runtime allocator/staging blits from input
#                             setup, not operator work.  They are classified as
#                             `runtime`, EXCLUDED from epoch segmentation and
#                             from the strips, and reported.  Left in, they cut
#                             the trace into 718 fragments and no epoch held
#                             both a GEMM and a collective.
RULES = [
    ("runtime", [r"^__amd_rocclr_", r"^__amd_"]),
    ("xgmi", [r"^nccl", r"^rccl", r"rccl", r"^mori", r"nccl_",
              r"reduce_?scatter", r"all_?reduce", r"all_?gather",
              r"sendrecv", r"reducescatter"]),
    ("mfma", [r"gemm", r"mfma", r"^cijk", r"_cijk", r"matmul", r"hgemm",
              r"bgemm", r"tensile"]),
    ("hbm",  [r"scatter", r"quant", r"combine", r"copy", r"memcpy",
              r"elementwise", r"vectorized_elementwise", r"add", r"cast",
              r"contiguous", r"fill", r"zero"]),
]

# Classes that are operator work and belong on a strip.
STRIP_CLASSES = ("mfma", "hbm", "xgmi")


def classify(name):
    low = name.lower()
    for klass, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, low):
                return klass, pattern
    return None, None


def find_trace(trace_dir):
    hits = sorted(glob.glob(os.path.join(trace_dir, "**", "*kernel_trace*.csv"),
                            recursive=True))
    if not hits:
        raise SystemExit(f"no *kernel_trace*.csv under {trace_dir}")
    # rocprofv3 writes one tree per traced process; rank 0 is the only one
    # wrapped, so a single file is expected.  If there are several, take the
    # largest and say so rather than merging streams from different processes.
    if len(hits) > 1:
        print(f"NOTE: {len(hits)} trace files; using the largest")
        hits.sort(key=os.path.getsize)
    return hits[-1]


def load_rows(path):
    with open(path, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fields = {f.lower().strip(): f for f in (reader.fieldnames or [])}

        def col(*candidates):
            for c in candidates:
                if c in fields:
                    return fields[c]
            raise SystemExit(f"trace {path} has no column among {candidates}; "
                             f"saw {list(fields)}")

        name_c = col("kernel_name", "name")
        start_c = col("start_timestamp", "start", "begin_timestamp")
        end_c = col("end_timestamp", "end")
        rows = []
        for row in reader:
            try:
                rows.append((row[name_c].strip(),
                             int(row[start_c]), int(row[end_c])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r[1])
    return rows


def segment(rows, min_gap_ns):
    """Split the dispatch stream into epochs on inter-kernel gaps."""
    epochs, current, last_end = [], [], None
    for row in rows:
        if last_end is not None and row[1] - last_end > min_gap_ns:
            epochs.append(current)
            current = []
        current.append(row)
        last_end = max(last_end or 0, row[2])
    if current:
        epochs.append(current)
    return epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--boundaries", default=None)
    ap.add_argument("--map-json", required=True)
    ap.add_argument("--events-json", required=True)
    ap.add_argument("--shape", required=True, help="m,n,k,bias,seed")
    ap.add_argument("--min-gap-us", type=float, default=40.0)
    args = ap.parse_args()

    m, n, k, bias, seed = [int(x) for x in args.shape.split(",")]
    path = find_trace(args.trace_dir)
    rows = load_rows(path)
    print(f"trace {path}: {len(rows)} dispatches")

    # ---- 1. the map, derived from the observed names ----------------------
    per_name = {}
    for name, start, end in rows:
        entry = per_name.setdefault(name, {"count": 0, "ns": 0})
        entry["count"] += 1
        entry["ns"] += end - start
    mapping, unclassified = {}, {}
    for name, entry in per_name.items():
        klass, rule = classify(name)
        target = mapping if klass else unclassified
        target[name] = dict(entry, resource=klass, matched_rule=rule)
    if unclassified:
        print(f"WARNING: {len(unclassified)} kernel names matched no rule; "
              f"they are recorded as `unclassified` and reported in result.md:")
        for name, entry in sorted(unclassified.items(),
                                  key=lambda kv: -kv[1]["ns"])[:10]:
            print(f"  {entry['ns']/1e3:10.1f} us  x{entry['count']:<5} {name[:90]}")

    with open(args.map_json, "w") as handle:
        json.dump({
            "schema": "exp22.b0_kernel_map.v1",
            "source_trace": path,
            "rules": {klass: pats for klass, pats in RULES},
            "classified": mapping,
            "unclassified": unclassified,
        }, handle, indent=2, sort_keys=True)
    print(f"wrote {args.map_json}")

    # ---- 2. epochs --------------------------------------------------------
    # Segment on operator dispatches only. The allocator blits are real GPU work
    # but they are the harness setting up inputs, not the operator under study,
    # and they are scattered densely enough to hide every real epoch boundary.
    op_rows = [r for r in rows if classify(r[0])[0] in STRIP_CLASSES]
    runtime_n = len(rows) - len(op_rows)
    print(f"segmentation input: {len(op_rows)} operator dispatches "
          f"({runtime_n} runtime/allocator dispatches excluded)")
    epochs = segment(op_rows, int(args.min_gap_us * 1000))
    spans = [(e[0][1], max(r[2] for r in e), len(e)) for e in epochs]
    print(f"segmented into {len(epochs)} epochs at a {args.min_gap_us} us gap")

    # Keep only epochs that contain both a GEMM-class and an xGMI-class
    # dispatch: warmup fragments and teardown do not.
    def complete(epoch):
        classes = {classify(r[0])[0] for r in epoch}
        return "mfma" in classes and "xgmi" in classes

    usable = [e for e in epochs if complete(e)]
    if not usable:
        raise SystemExit("no epoch contains both a GEMM and a collective "
                         "dispatch -- check the classification rules against "
                         "the names printed above")
    # Drop the first usable epoch (first-touch effects) and take the median by
    # duration, which is the same statistic the rest of the night reports.
    candidates = usable[1:] or usable
    durations = [max(r[2] for r in e) - e[0][1] for e in candidates]
    pick = durations.index(statistics.median_low(durations))
    chosen, t0 = candidates[pick], candidates[pick][0][1]
    print(f"usable epochs: {len(usable)}; chosen duration "
          f"{durations[pick]/1e3:.1f} us over {len(chosen)} dispatches")

    host = None
    if args.boundaries and os.path.exists(args.boundaries):
        with open(args.boundaries) as handle:
            host = json.load(handle)
        print(f"host cross-check: median device_us "
              f"{host.get('device_us_median'):.1f} vs traced epoch "
              f"{durations[pick]/1e3:.1f} us")

    intervals = []
    for name, start, end in chosen:
        klass, _ = classify(name)
        intervals.append({
            "name": name,
            "resource": klass or "unclassified",
            "begin_ns": start - t0,
            "end_ns": end - t0,
        })

    with open(args.events_json, "w") as handle:
        json.dump({
            "schema": "exp22.events.v1",
            "arm": "b0_reference",
            "kind": "kernel_trace",
            "shape": {"m": m, "n": n, "k": k, "has_bias": bool(bias),
                      "seed": seed},
            "epoch_ns": durations[pick],
            "epoch_count_traced": len(epochs),
            "epoch_count_usable": len(usable),
            "runtime_dispatches_excluded": runtime_n,
            "epoch_durations_ns": durations,
            "intervals": intervals,
            "unclassified_names": sorted(unclassified),
            "host_boundaries": host,
            "source_trace": path,
            "note": "rank 0 only; the other seven ranks ran unprofiled so the "
                    "collective had real peers. Timestamps are ns relative to "
                    "the first dispatch of the chosen epoch.",
        }, handle, indent=2)
    print(f"wrote {args.events_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
