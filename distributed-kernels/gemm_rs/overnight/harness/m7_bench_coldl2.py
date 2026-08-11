"""exp_07: is the evaluator/harness gap on shapes 4 and 6 a cold-L2 effect?

A COPY of m7_bench.py's single-shot path with one thing added: the official
evaluator's `clear_l2_cache()` run before each timed iteration. Nothing under
harness/ is edited; this is a new file.

The evaluator (utils.py:169) is verbatim:

    def clear_l2_cache():
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy

i.e. a 256 MiB write on the calling rank's own device. It is called at
eval.py:349, *before* `torch.cuda.synchronize(); dist.barrier(); t0`, so the
flush is not itself inside the timed region -- only its effect is. This file
reproduces that placement exactly.

Because the evaluator runs eight rank processes and each flushes its own
device, the single-process harness must flush all eight.

Arms are measured PAIRED: within one process, one GemmRS instance, one shape,
alternating blocks with the block order flipped every rep so neither arm is
systematically first or last. Every arm reports mean/median/stdev/min/max.
"""

import json
import math
import statistics
import sys
import time

import torch

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def clear_l2_cache_all(mib):
    """The evaluator's clear_l2_cache(), applied to every device.

    `mib=256` is byte-identical to the evaluator's (32,1024,1024) int64 tensor.
    Smaller sizes are a control: MI300X has 4 MB of L2 per XCD (32 MB total)
    but 256 MB of Infinity Cache, so sweeping the size separates the two.
    """
    elems = (mib * 1024 * 1024) // 8
    for rank in range(WORLD):
        torch.cuda.set_device(rank)
        dummy = torch.empty(elems, dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy


def _stats(samples):
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def single_shot(h, iters, flush_mib):
    """One `barrier -> t0 -> call -> synchronize -> t1` iteration, `iters` times.

    Identical to harness_lib.time_single_shot's inner loop; the only addition is
    the optional flush, which is placed before the timing barrier exactly as the
    evaluator places it.
    """
    walls, devs, flushes = [], [], []
    for _ in range(iters):
        h.sync()
        if flush_mib:
            flush_start = time.perf_counter()
            clear_l2_cache_all(flush_mib)
            h.sync()
            flushes.append((time.perf_counter() - flush_start) * 1e6)
        starts, ends = h._events()
        wall_start = time.perf_counter()
        for rank in range(WORLD):
            torch.cuda.set_device(rank)
            starts[rank].record(torch.cuda.current_stream(rank))
            h._launch_one(rank)
            ends[rank].record(torch.cuda.current_stream(rank))
        h.n_calls += 1
        h.sync()
        walls.append((time.perf_counter() - wall_start) * 1e6)
        devs.append(max(starts[r].elapsed_time(ends[r]) * 1e3
                        for r in range(WORLD)))
    return walls, devs, flushes


def measure(shape, arms, iters, reps, warmup_ms):
    m, n, k, bias, seed = shape
    with GemmRS(m, n, k, bias) as h:
        h.set_inputs(seed)
        h.launch()
        checks = h.verify()
        tight = h.verify(rtol=2e-3, atol=2e-3)

        # Duration-based warmup: a fixed iteration count warms a fast shape for
        # only a few ms and measures a GPU that is still ramping.
        warmup_start = time.perf_counter()
        while (time.perf_counter() - warmup_start) * 1e3 < warmup_ms:
            for _ in range(8):
                h.launch(sync=False)
            h.sync()

        walls = {a: [] for a in arms}
        devs = {a: [] for a in arms}
        flushes = {a: [] for a in arms}
        for rep in range(reps):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                w, d, f = single_shot(h, iters, arm)
                walls[arm].extend(w)
                devs[arm].extend(d)
                flushes[arm].extend(f)

        post = h.verify(rtol=2e-3, atol=2e-3)
        return {
            "correct": all(c["allclose"] for c in checks),
            "correct_tight": all(c["allclose"] for c in tight),
            "correct_after": all(c["allclose"] for c in post),
            "max_abs_diff": max(c["max_abs_diff"] for c in checks),
            "errors": h.error_report(),
            "epoch_problems": h.check_epochs()[:2],
            "wall": {a: _stats(walls[a]) for a in arms},
            "dev": {a: _stats(devs[a]) for a in arms},
            "flush": {a: (_stats(flushes[a]) if flushes[a] else None)
                      for a in arms},
            "plan": {key: h.plan[key] for key in
                     ("config_row", "bm", "bn", "bk", "eb", "num_reducer_ctas",
                      "gemm_tiles", "red_tiles", "c_heap_bytes")},
        }


def main():
    rt.enable_peer_access(WORLD)
    # argv: arms(csv MiB, 0 == warm) iters reps shapes(csv 1-based, or "all")
    arms = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                             else "0,256").split(",")]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    which = sys.argv[4] if len(sys.argv) > 4 else "all"
    tag = sys.argv[5] if len(sys.argv) > 5 else "main"
    indices = (list(range(len(SCORED))) if which == "all"
               else [int(x) - 1 for x in which.split(",")])
    warmup_ms = 400

    print(f"arms(MiB flush) = {arms}  iters/arm/rep = {iters}  reps = {reps}")
    print(f"shapes = {[i + 1 for i in indices]}")

    detail = {}
    for index in indices:
        shape = SCORED[index]
        result = measure(shape, arms, iters, reps, warmup_ms)
        detail[index] = result
        ok = (result["correct_tight"] and result["correct_after"]
              and not result["errors"] and not result["epoch_problems"])
        print(f"\n--- shape {index+1}: {shape[0]}x{shape[1]}x{shape[2]} "
              f"bias={int(shape[3])}  heap={result['plan']['c_heap_bytes']/1e6:.1f} MB "
              f"tiles={result['plan']['gemm_tiles']} "
              f"{'ok' if ok else 'INVALID: ' + str(result['errors']) + str(result['epoch_problems'])}")
        base = result["wall"][arms[0]]["mean"]
        for arm in arms:
            w = result["wall"][arm]
            d = result["dev"][arm]
            f = result["flush"][arm]
            label = "warm" if arm == 0 else f"cold{arm}"
            flush_cost = "-" if f is None else "%.0f" % f["mean"]
            print(f"    {label:>8}  wall mean={w['mean']:9.2f} "
                  f"med={w['median']:9.2f} sd={w['stdev']:7.2f} "
                  f"({100*w['stdev']/w['mean']:4.1f}%) "
                  f"min={w['min']:9.2f} max={w['max']:9.2f} | "
                  f"dev_max mean={d['mean']:9.2f} | "
                  f"x_warm={w['mean']/base:5.2f} | "
                  f"flush_cost={flush_cost} us")

    print("\n" + "=" * 118)
    print("exp_07 cold-L2: single-shot wall (us), paired, same process/instance")
    print("=" * 118)
    header = f"{'#':>2}{'m':>6}{'n':>6}{'k':>7}"
    for arm in arms:
        label = "warm" if arm == 0 else f"cold{arm}"
        header += f"{label + ' mean':>14}{'sd%':>7}"
    header += f"{'ratio':>8}{'ok':>4}"
    print(header)
    print("-" * len(header))
    per_arm_means = {a: [] for a in arms}
    for index in indices:
        shape = SCORED[index]
        result = detail[index]
        row = f"{index+1:>2}{shape[0]:>6}{shape[1]:>6}{shape[2]:>7}"
        for arm in arms:
            w = result["wall"][arm]
            per_arm_means[arm].append(w["mean"])
            row += f"{w['mean']:>14.2f}{100*w['stdev']/w['mean']:>7.1f}"
        ratio = (result["wall"][arms[-1]]["mean"]
                 / result["wall"][arms[0]]["mean"])
        ok = (result["correct_tight"] and result["correct_after"]
              and not result["errors"] and not result["epoch_problems"])
        row += f"{ratio:>8.2f}{'yes' if ok else 'NO':>4}"
        print(row)
    print("-" * len(header))
    for arm in arms:
        label = "warm" if arm == 0 else f"cold{arm} MiB"
        print(f"geomean ({label:>12}) : {geomean(per_arm_means[arm]):9.2f} us")

    out = f"coldl2_results_{tag}.json"
    with open(out, "w") as handle:
        json.dump({
            "arms_mib": arms, "iters": iters, "reps": reps,
            "shapes": [SCORED[i] for i in indices],
            "detail": {str(k): v for k, v in detail.items()},
            "geomean_by_arm": {str(a): geomean(per_arm_means[a])
                               for a in arms},
        }, handle, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
