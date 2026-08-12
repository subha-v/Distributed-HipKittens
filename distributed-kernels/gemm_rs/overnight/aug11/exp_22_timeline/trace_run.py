#!/usr/bin/env python3
"""exp_22 arm (b): drive the flag-ON megakernel and dump the phase-event ring.

DIAGNOSTIC ARM.  No microsecond produced by this script is a performance
number.  The only timing it publishes is the ON-vs-OFF perturbation estimate,
which is labelled as such.

harness/ is READ-ONLY here.  `harness_lib.GemmRS` is imported and used
unmodified; the trace module is loaded from exp_22_timeline/build/ and swapped
in after construction, and the ring pointer is appended to the prebuilt
argument tuples rather than by editing `_rebuild_args`.  Nothing is written
into harness/build/.

Tick calibration is in-situ and needs no extra launch: for every traced launch
we record rank 0's device span in ticks, `max(CTA_END) - min(CTA_BEG)`, against
the same launch's hipEvent device time.  Across shapes of very different
durations the slope of that line is ticks/us and the intercept absorbs the
fixed offset between dispatch and the first CTA's stamp -- which is exactly the
bias a single-point ratio would fold into the rate.

  trace_run.py [--shapes s5,s6] [--launches 3] [--outdir .]
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ON, "harness"))

import torch                              # noqa: E402
import harness_lib as H                   # noqa: E402

rt = H.rt

CU_COUNT = 304
DEPTH = 64                                # must equal hk_trace::DEPTH
USABLE = DEPTH - 1

# exp_21's independent calibration of the same instruction on this node:
# 99.7366 MHz = 9.97366e-2 ticks/ns = 99.7366 ticks/us, 5 reps, spread 0.0101%,
# measured two-stage against steady_clock in a dedicated kernel
# (aug11/exp_21_saturation/saturation.json:tick_rate_hz, LESSONS.md:1051).
# It shares no machinery with the in-situ regression below beyond the
# instruction itself, which is what makes the agreement meaningful.
EXP21_TICKS_PER_US = 99.7358085
PHASE_NAMES = [
    "CTA_BEG", "MAINLOOP_BEG", "MAINLOOP_END", "CREDIT_WAIT_BEG",
    "CREDIT_WAIT_END", "EMIT_BEG", "EMIT_END", "RELEASE_BEG", "RELEASE_END",
    "PUBLISH_END", "READY_WAIT_BEG", "READY_WAIT_END", "REDUCE_BEG",
    "REDUCE_END", "CREDIT_PUB_END", "CTA_END",
]

# tag -> (m, n, k, has_bias, seed); seeds are cases_bench.txt's, so every arm
# of every experiment shares the size AND the input distribution.
SHAPES = {
    "s1": (64, 7168, 18432, False, 1234),
    "s2": (512, 4096, 12288, True, 663),
    "s3": (2048, 2880, 2880, True, 166),
    "s4": (4096, 4096, 4096, False, 1371),
    "s5": (8192, 4096, 14336, True, 7168),
    "s6": (8192, 8192, 29568, False, 42),
}


def load_trace_module():
    path = os.path.join(HERE, "build", "gemm_rs_mi300x_trace.so")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run build_trace.sh first")
    spec = importlib.util.spec_from_file_location("gemm_rs_mi300x_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_ring(raw):
    """bytes -> {cta: [(phase_id, ticks), ...]}, plus the drop count."""
    events, drops = {}, 0
    for cta in range(CU_COUNT):
        base = cta * DEPTH * 8
        slots = []
        for slot in range(USABLE):
            off = base + slot * 8
            word = int.from_bytes(raw[off:off + 8], "little")
            if word == 0:
                break
            slots.append((word >> 56, word & 0x00FFFFFFFFFFFFFF))
        drop_word = int.from_bytes(
            raw[base + USABLE * 8: base + DEPTH * 8], "little")
        drops += drop_word
        if slots:
            events[cta] = slots
    return events, drops


def run_shape(tag, module, launches, warmup_ms):
    m, n, k, has_bias, seed = SHAPES[tag]
    print(f"\n===== {tag}: {m}x{n}x{k} bias={has_bias} seed={seed} =====",
          flush=True)

    inst = H.GemmRS(m, n, k, has_bias)
    # Swap in the diagnostic module AFTER construction: __init__ loads the
    # production .so from harness/build (read-only), and we never write there.
    inst.module = module
    inst.entry = module.gemm_rs_mi300x

    ring_bytes = CU_COUNT * DEPTH * 8
    rings = [rt.plain_alloc(r, ring_bytes) for r in range(H.WORLD)]
    try:
        inst.set_inputs(seed)
        for rank in range(H.WORLD):
            inst._args[rank] = inst._args[rank] + (rings[rank],)
        # Pin the ctrl key so harness_lib's launch path cannot rebuild the
        # tuples and drop the ring pointer we just appended.
        inst._ctrl_key = (0, 0, 0)

        start = time.perf_counter()
        while (time.perf_counter() - start) * 1e3 < warmup_ms:
            for _ in range(8):
                inst.launch(sync=False)
            inst.sync()
        errors = inst.error_report()
        if errors:
            raise SystemExit(f"{tag}: protocol error bits after warmup: {errors}")

        captures = []
        for index in range(launches):
            for rank in range(H.WORLD):
                rt.fill_bytes(rank, rings[rank], 0, ring_bytes)
            inst.sync()

            starts, ends = inst._events()
            for rank in range(H.WORLD):
                torch.cuda.set_device(rank)
                starts[rank].record(torch.cuda.current_stream(rank))
            inst.launch(sync=False)
            for rank in range(H.WORLD):
                torch.cuda.set_device(rank)
                ends[rank].record(torch.cuda.current_stream(rank))
            inst.sync()
            device_us = [starts[r].elapsed_time(ends[r]) * 1e3
                         for r in range(H.WORLD)]

            per_rank = {}
            for rank in range(H.WORLD):
                raw = rt.device_to_host(rank, rings[rank], ring_bytes)
                events, drops = decode_ring(raw)
                per_rank[rank] = (events, drops)
            captures.append((device_us, per_rank))
            print(f"  launch {index}: device_us max={max(device_us):.1f} "
                  f"rank0={device_us[0]:.1f} drops={per_rank[0][1]}", flush=True)

        # Correctness at BOTH tolerances -- a diagnostic build that computes a
        # different answer is picturing a different kernel.
        checks = {}
        for label, tol in (("1e-2", 1e-2), ("2e-3", 2e-3)):
            results = inst.verify(rtol=tol, atol=tol)
            checks[label] = {
                "all_close": all(r["allclose"] for r in results),
                "max_abs_diff": max(r["max_abs_diff"] for r in results),
            }
            print(f"  correctness {label}: {checks[label]}", flush=True)
        problems = inst.check_epochs() + inst.check_signals()
        if problems:
            print("  WARNING protocol state: " + "; ".join(problems[:4]))

        plan = {key: (int(value) if isinstance(value, (int, bool)) else value)
                for key, value in inst.plan.items()}
        return {
            "tag": tag,
            "shape": {"m": m, "n": n, "k": k, "has_bias": has_bias,
                      "seed": seed},
            "plan": plan,
            "captures": captures,
            "correctness": checks,
            "protocol_problems": problems,
        }
    finally:
        for rank in range(H.WORLD):
            rt.free_device(rank, rings[rank])
        inst.close()


def span_ticks(events):
    begins = [t for slots in events.values() for pid, t in slots if pid == 0]
    ends = [t for slots in events.values() for pid, t in slots if pid == 15]
    if not begins or not ends:
        return None
    return max(ends) - min(begins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="s5,s6,s2",
                    help="s5 is the figure; s6 and s2 give the tick "
                         "regression two more, very different, durations")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup-ms", type=float, default=800.0)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    rt.enable_peer_access(H.WORLD)
    module = load_trace_module()
    tags = [t.strip() for t in args.shapes.split(",") if t.strip()]

    results, calib = {}, []
    for tag in tags:
        result = run_shape(tag, module, args.launches, args.warmup_ms)
        results[tag] = result
        for device_us, per_rank in result["captures"]:
            span = span_ticks(per_rank[0][0])
            if span:
                calib.append({"tag": tag, "span_ticks": span,
                              "device_us_rank0": device_us[0]})

    # ---- tick rate: slope of span_ticks vs device_us across all points -----
    ticks_per_us, intercept, r2 = None, None, None
    if len(calib) >= 2:
        xs = [c["device_us_rank0"] for c in calib]
        ys = [float(c["span_ticks"]) for c in calib]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            ticks_per_us = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            intercept = my - ticks_per_us * mx
            ss_res = sum((y - (ticks_per_us * x + intercept)) ** 2
                         for x, y in zip(xs, ys))
            ss_tot = sum((y - my) ** 2 for y in ys)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    tick = {
        "schema": "exp22.tick_rate.v1",
        "method": "in-situ regression of rank-0 device span (ticks) on hipEvent "
                  "device time (us) across shapes; slope = ticks/us",
        "points": calib,
        "ticks_per_us": ticks_per_us,
        "intercept_ticks": intercept,
        "r2": r2,
        "sibling_hypothesis_ticks_per_us": 100.0,
        "exp21_measured_ticks_per_us": EXP21_TICKS_PER_US,
        "exp21_spread_pct": 0.0101,
        "note": "The gfx950 sibling asserts s_memrealtime is 100 MHz (10 ns "
                "ticks). FIGURE_SPECS.md section 5 item 7 flags that as a "
                "quantity to verify on gfx942. The number used by the figure "
                "is the measured slope above; the hypothesis is recorded only "
                "so the two can be compared in result.md.",
    }
    # Two independent calibrations agreeing is worth far more than one, and a
    # disagreement means this figure's x-axis is wrong -- a bad tick rate does
    # not distort the plot visibly, it silently rescales the whole time axis.
    # exp_21 measured the rate directly against steady_clock in a dedicated
    # kernel; this arm regresses the production kernel's own span against
    # hipEvent time, so the two share no machinery beyond the instruction.
    if ticks_per_us:
        for label, reference in (("exp_21 sat_calib", EXP21_TICKS_PER_US),
                                 ("sibling gfx950 spec", 100.0)):
            disagreement = abs(ticks_per_us - reference) / reference
            verdict = "agree" if disagreement <= 0.01 else "DISAGREE"
            tick.setdefault("cross_checks", []).append({
                "reference": label, "ticks_per_us": reference,
                "rel_delta": disagreement, "verdict": verdict})
            print(f"  vs {label:<22} {reference:9.4f}: "
                  f"{disagreement:+.3%}  {verdict}")
    with open(os.path.join(args.outdir, "tick_rate.json"), "w") as handle:
        json.dump(tick, handle, indent=2)
    print(f"\nticks_per_us = {ticks_per_us} (r2={r2}); "
          f"sibling hypothesis 100.0")

    # ---- events.json, one per shape; the figure uses s5 -------------------
    for tag, result in results.items():
        device_us, per_rank = result["captures"][-1]
        payload = {
            "schema": "exp22.events.v1",
            "arm": "ours",
            "kind": "phase_ring",
            "diagnostic_arm": True,
            "diagnostic_policy": "HK_GEMM_RS_MI300X_TRACE is OFF for all "
                                 "campaign timing; no microsecond in this file "
                                 "is a performance number.",
            "shape": result["shape"],
            "plan": result["plan"],
            "ring": {"cu_count": CU_COUNT, "depth": DEPTH,
                     "usable_slots": USABLE,
                     "entry": "(phase_id << 56) | (s_memrealtime() & 0x00FFFFFFFFFFFFFF)"},
            "phase_names": PHASE_NAMES,
            "ticks_per_us": ticks_per_us,
            "device_us": {"per_rank": device_us,
                          "rank0": device_us[0], "max": max(device_us)},
            "correctness": result["correctness"],
            "protocol_problems": result["protocol_problems"],
            "ranks": {
                str(rank): {
                    "drops": drops,
                    "ctas_seen": len(events),
                    "aborted_ctas": sum(
                        1 for slots in events.values()
                        if not any(pid == 15 for pid, _ in slots)),
                    "events": [[cta, pid, tick_value]
                               for cta, slots in sorted(events.items())
                               for pid, tick_value in slots],
                }
                for rank, (events, drops) in sorted(per_rank.items())
            },
        }
        path = os.path.join(args.outdir, f"events_ours_{tag}.json")
        with open(path, "w") as handle:
            json.dump(payload, handle)
        rank0 = payload["ranks"]["0"]
        print(f"wrote {path}: {len(rank0['events'])} rank-0 events, "
              f"{rank0['ctas_seen']} CTAs, drops={rank0['drops']}, "
              f"aborted={rank0['aborted_ctas']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
