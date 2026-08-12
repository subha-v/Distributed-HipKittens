"""E4b: per-shape BM/BN/BK screening sweep with an explicit null arm.

Same protocol as exp_13's NR sweep, which is what established that sweep's two
winners and rejected its two false positives:

  * every arm is a separate, independently zeroed GemmRS, and all arms for a
    shape are live at once so passes can interleave them;
  * operand tensors are generated ONCE per shape and shared by every arm, so
    arms differ only in their tile and in their own output heap;
  * complete rotation: passes == len(arms), order rotated by one per pass, so
    every arm occupies every position exactly once and the per-position residual
    is reported as a direct measurement of the positional component;
  * the landed tile is carried by TWO independent arms (the null twin). Their
    spread is the floor of this instrument, because the documented bias on this
    node is PER-ALLOCATION, not positional -- two identical arms have measured
    4.28% apart on shape 6 while the positional residual stayed inside 0.47%;
  * best (min) and median only. Means are unusable here.

ONE STRUCTURAL DIFFERENCE FROM exp_13, and it is the load-bearing one.

`num_reducer_ctas` is a host scalar (dhk_rt.cpp:197), so one build served every
arm of that sweep. BM/BN/BK are TEMPLATE parameters, so each candidate needs its
own instantiation; they live at config_row 101-106 in a separate module built
with -DHK_GEMM_RS_MI300X_TILE_SWEEP=1, and the resolved plan is overridden on
the Python side to match.

That re-derivation is the one place this instrument could lie -- a wrong
lrow_count or ready_words would mis-size the signal heap and look like a
performance result. So it is validated rather than trusted: check_derivation()
asserts that deriving the LANDED tile through the Python path reproduces
rt.resolve_shape's dict field for field on all six shapes, and main() refuses to
run if any field differs. Every arm additionally asserts that its derived
even_k equals the K_TAIL its config_row was compiled with, so a mismatched
(row, tail) pair cannot silently run.

usage: sweep.py [shapes] [passes] [iters_scale]
   e.g. sweep.py 1,2,3 0 1        <- 0 passes => one complete rotation
        sweep.py 1 8 2
"""

import contextlib
import json
import math
import os
import statistics
import sys
import time
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import torch                                                    # noqa: E402
import harness_lib as H                                         # noqa: E402
from harness_lib import rt, WORLD, GemmRS                       # noqa: E402

CU = 304
SWEEP_MODULE = "gemm_rs_mi300x_tilesweep"
# Reverse the order in which the arms are CONSTRUCTED (and therefore the order
# in which they draw from hipMalloc). Does not change the timing rotation.
REVERSE = os.environ.get("SWEEP_REVERSE") == "1"

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]
# Chosen so one timed block is ~25-45 ms of device work on every shape.
ITERS = [300, 250, 250, 120, 50, 25]
PRIME = 4
TIGHT = 2e-3
GRADED = 1e-2

# (config_row, BM, BN, BK, expected K_TAIL) for the screening instantiations.
# MUST agree with the guarded rows of dispatch_gemm_rs_mi300x.
SWEEP_ROWS = {
    101: (32, 64, 128, False),
    102: (32, 64, 160, True),
    103: (64, 128, 64, False),
    104: (64, 64, 128, False),
    105: (128, 192, 32, True),
    106: (64, 192, 64, True),
    107: (64, 192, 64, False),
}

# Arms per shape index. "T" is the landed tile taken from scored_shapes.
#
# Row 104 (<64,64,128>) is retained even though draw 1 disqualified it on
# correctness: it is the pre-registered discriminator for shape 2, and a
# reproducible failure with a diagnosis is a result. diagnose_dead() below
# characterises it instead of the timing loop.
ARMS = {
    0: ["T", 101, 102],
    1: ["T", 103, 104, 107],
    2: ["T", 105, 106],
    3: ["T"],
    4: ["T"],
    5: ["T"],
}


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ---------------------------------------------------------------------------
# The derivation. Every line mirrors host_abi.hpp::resolve_shape, in its order,
# including the gates -- a candidate that would have thrown there must throw
# here rather than run with a quietly different geometry.
# ---------------------------------------------------------------------------
def derive(m, n, k_global, has_bias, bm, bn, bk, config_row, nr):
    if m <= 0 or n <= 0 or k_global <= 0:
        raise ValueError("non-positive dimension")
    if m % WORLD or k_global % WORLD:
        raise ValueError("M or K not divisible by world")
    k_local = k_global // WORLD
    slice_rows = m // WORLD
    eb = gcd(bm, slice_rows)
    if eb <= 0 or slice_rows % eb or bm % eb:
        raise ValueError("emit band does not divide")
    if bm % 32 or bn % 64:                       # WARPS_M*16, WARPS_N*16
        raise ValueError("warp fragment mismatch")
    lrow_count = slice_rows // eb
    col_count = (n + bn - 1) // bn
    if m % bm:
        raise ValueError("BM must divide M")
    gemm_tiles = (m // bm) * col_count
    red_tiles = lrow_count * col_count
    if lrow_count > 32 or col_count > 128:
        raise ValueError("signal geometry exceeds LROW_MAX/COL_MAX")
    num_gemm_ctas = CU - nr
    if num_gemm_ctas <= 0 or num_gemm_ctas > CU:
        raise ValueError("degenerate CTA split")
    ready_words = WORLD * lrow_count * col_count
    signal_words_total = rt.SIGNAL_GUARD_U32 + 2 * ready_words
    if signal_words_total > 65600:
        raise ValueError("signal heap too small")
    lds_bytes = 2 * (bm + bn) * bk * 2
    if lds_bytes > 65536:
        raise ValueError(f"LDS {lds_bytes} > 65536")
    if 32 * bn * 2 > lds_bytes:
        raise ValueError("stage window > LDS")
    if bk % 32:
        raise ValueError("KS % 16 != 0")
    if bm * bk > 65536 or bn * bk > 65536:
        raise ValueError("g2s big_calls > 1")
    return {
        "m": m, "n": n, "k_local": k_local, "has_bias": has_bias,
        "bm": bm, "bn": bn, "bk": bk,
        "num_reducer_ctas": nr, "config_row": config_row,
        "slice_rows": slice_rows, "eb": eb,
        "lrow_count": lrow_count, "col_count": col_count,
        "gemm_tiles": gemm_tiles, "red_tiles": red_tiles,
        "num_gemm_ctas": num_gemm_ctas,
        "c_heap_bytes": WORLD * slice_rows * n * 2,
        "ready_words": ready_words, "credit_words": ready_words,
        "signal_words_total": signal_words_total,
        "even_k": (k_local % bk) == 0,
        "even_n": (n % bn) == 0,
        "packet_fast_path": (n % 8) == 0,
        "lds_bytes": lds_bytes,
    }


def check_derivation():
    """derive() on the LANDED tile must reproduce resolve_shape field for field.

    This is the whole warrant for overriding the plan in Python. If it passes,
    the only thing an override changes is the tile; if it fails, every number
    this script would print is suspect and it must not run.
    """
    problems = []
    for m, n, k, bias, _seed in SCORED:
        want = dict(rt.resolve_shape(m, n, k, bias))
        got = derive(m, n, k, bias, int(want["bm"]), int(want["bn"]),
                     int(want["bk"]), int(want["config_row"]),
                     int(want["num_reducer_ctas"]))
        if set(got) != set(want):
            problems.append(f"{m}x{n}x{k}: key sets differ "
                            f"{sorted(set(want) ^ set(got))}")
            continue
        for key in sorted(want):
            if int(got[key]) != int(want[key]):
                problems.append(f"{m}x{n}x{k}: {key} derived={got[key]} "
                                f"resolve_shape={want[key]}")
    return problems


@contextlib.contextmanager
def forced_plan(plan):
    """Make GemmRS.__init__ resolve to `plan`.

    The plan has to be in place BEFORE the constructor runs, because the
    symmetric signal heap is sized from signal_words_total, which the tile
    moves. Scoped to one construction and restored unconditionally.
    """
    original = rt.resolve_shape
    rt.resolve_shape = lambda *a, **kw: dict(plan)
    try:
        yield
    finally:
        rt.resolve_shape = original


class Arm:
    def __init__(self, label, plan, handle):
        self.label = label
        self.plan = plan
        self.h = handle
        self.dead = None
        self.samples = []          # (pass, position, wall_us, device_us)

    @property
    def walls(self):
        return [s[2] for s in self.samples]

    @property
    def devs(self):
        return [s[3] for s in self.samples]


def diagnose_dead(arm, want):
    """Separate "the kernel ran and computed the wrong thing" from "it never ran".

    These look identical through a tolerance check -- an output left at its
    initial zeros reports max|diff| == max|ref| -- but they mean opposite things.
    The epoch cell is the discriminator: it is incremented by every scheduled
    CTA at entry, before any protocol step, so cells still at 0 after a launch
    prove the launch itself did not happen (a failed dynamic-LDS reservation
    raises no exception here -- the entry point does not check hipGetLastError).
    """
    zero_ranks = sum(1 for r in range(WORLD)
                     if not bool(arm.h.out[r].any()))
    cells = arm.h.epoch_cells(0)
    prod = cells[:int(arm.plan["num_gemm_ctas"])]
    red = cells[rt.EP_RED_U32:
                rt.EP_RED_U32 + int(arm.plan["num_reducer_ctas"])]
    ref_absmax = max(float(want[r].float().abs().max()) for r in range(WORLD))
    note = (f"    diag: out all-zero on {zero_ranks}/{WORLD} ranks; "
            f"rank0 epoch cells producer max={max(prod)} reducer max={max(red)}"
            f" (expect {arm.h.n_calls}); max|ref|={ref_absmax:.2e}")
    if zero_ranks == WORLD and max(prod) == 0:
        note += "\n    => THE LAUNCH NEVER RAN (not a numerics bug)"
    elif max(prod) == arm.h.n_calls:
        note += "\n    => kernel ran to completion and computed wrong values"
    return note


def cost_columns(plan):
    """The three pre-registered model columns, from the plan alone."""
    ng = int(plan["num_gemm_ctas"])
    tiles = int(plan["gemm_tiles"])
    waves = (tiles + ng - 1) // ng
    k_iters = (int(plan["k_local"]) + int(plan["bk"]) - 1) // int(plan["bk"])
    nr = int(plan["num_reducer_ctas"])
    return {
        "ng": ng, "waves": waves, "k_iters": k_iters,
        "rounds": (int(plan["red_tiles"]) + nr - 1) // nr,
        "last_fill": (tiles - (waves - 1) * ng) / ng,
        "wave_cost": waves * int(plan["bm"]) * int(plan["bn"]),
        "iter_cost": waves * k_iters,
        "traffic": 1.0 / int(plan["bm"]) + 1.0 / int(plan["bn"]),
    }


def build_arms(index, labels):
    m, n, k, bias, _seed = SCORED[index]
    landed = dict(rt.resolve_shape(m, n, k, bias))
    nr = int(landed["num_reducer_ctas"])
    specs = []
    for label in labels:
        if label == "T":
            specs.append(("T", landed))
        else:
            bm, bn, bk, want_tail = SWEEP_ROWS[label]
            plan = derive(m, n, k, bias, bm, bn, bk, label, nr)
            # The instantiation's K_TAIL is fixed at compile time; the plan's
            # even_k is derived independently. They must be complements.
            if plan["even_k"] == want_tail:
                raise SystemExit(
                    f"row {label} compiled with K_TAIL={want_tail} but "
                    f"K_local={plan['k_local']} % BK={bk} gives "
                    f"even_k={plan['even_k']} -- fix the dispatch row")
            specs.append((f"{bm}/{bn}/{bk}", plan))
    # The null twin is the landed tile a second time, appended at the far end so
    # the two copies are maximally separated in every rotation.
    specs.append(("T*", landed))
    return specs


def run_shape(index, passes, iters):
    m, n, k, bias, seed = SCORED[index]
    tag = f"{m}x{n}x{k}"
    inputs = [H.generate_input(r, WORLD, m, n, k, bias, seed)
              for r in range(WORLD)]
    ref = H.reference_reduce_scatter(inputs, m, n)
    want = [ref[r].to(f"cuda:{r}") for r in range(WORLD)]

    specs = build_arms(index, ARMS[index])
    # Draws 1-10 exposed something the null arm was built to catch but not to
    # explain: on shape 1 the twin contrast T-vs-T* was negative in 10 of 10
    # draws (median -0.91%), i.e. NOT symmetric noise. T is constructed first
    # and T* last, so allocation ORDER carries a systematic component on top of
    # the random per-allocation offset. Every candidate is constructed between
    # them, so that component biases candidates against T. Reversing the
    # construction order for half the draws both tests the mechanism (the twin
    # contrast should change sign) and makes the pooled null symmetric.
    if REVERSE:
        specs = list(reversed(specs))
    arms = []
    for label, plan in specs:
        with forced_plan(plan):
            handle = GemmRS(m, n, k, bias, module_name=SWEEP_MODULE)
        assert dict(handle.plan) == dict(plan), "plan did not take"
        handle.inputs = inputs                 # shared operands, same addresses
        handle._rebuild_args()
        arms.append(Arm(label, plan, handle))

    base = arms[0].plan
    bcost = cost_columns(base)
    print(f"\n===== shape {index + 1}: {tag} bias={int(bias)}  landed "
          f"{base['bm']}/{base['bn']}/{base['bk']} row={base['config_row']} "
          f"NR={base['num_reducer_ctas']} NG={bcost['ng']}  "
          f"iters={iters} passes={passes} arms={len(arms)} =====", flush=True)

    for a in arms:
        c = cost_columns(a.plan)
        a.cost = c
        a.h.launch()
        errs = a.h.error_report()
        worst = 0.0
        ok_tight = ok_graded = True
        for r in range(WORLD):
            got = a.h.out[r]
            worst = max(worst, float((got.float() - want[r].float()).abs().max()))
            ok_graded &= bool(torch.allclose(got, want[r], rtol=GRADED,
                                             atol=GRADED))
            ok_tight &= bool(torch.allclose(got, want[r], rtol=TIGHT,
                                            atol=TIGHT))
        a.max_abs_diff, a.ok_graded, a.ok_tight = worst, ok_graded, ok_tight
        if errs:
            a.dead = "errbit:" + ";".join(errs)
        elif not ok_graded:
            a.dead = f"WRONG@1e-2 max|diff|={worst:.2e}"
        elif not ok_tight:
            a.dead = f"WRONG@2e-3 max|diff|={worst:.2e}"
        print(f"  {a.label:<12} tiles={a.plan['gemm_tiles']:<5} "
              f"wav={c['waves']} fill={c['last_fill'] * 100:3.0f}% "
              f"ki={c['k_iters']:<4} tail={int(not a.plan['even_k'])} "
              f"cols={a.plan['col_count']:<4} red={a.plan['red_tiles']:<4} "
              f"rnd={c['rounds']:<3} LDS={a.plan['lds_bytes'] // 1024:>2}K "
              f"w*x={c['wave_cost'] / bcost['wave_cost']:.2f} "
              f"w*ki={c['iter_cost'] / bcost['iter_cost']:.2f} "
              f"traf={c['traffic'] / bcost['traffic']:.2f} "
              f"max|diff|={worst:.2e} "
              f"{'OK' if a.dead is None else 'DEAD ' + a.dead}", flush=True)
        if a.dead is not None:
            a.diag = diagnose_dead(a, want)
            print(a.diag, flush=True)

    live = [a for a in arms if a.dead is None]
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < 300 and live:
        for a in live:
            for _ in range(2):
                a.h.launch(sync=False)
            a.h.sync()

    for p in range(passes):
        rot = p % len(arms)
        order = arms[rot:] + arms[:rot]
        for position, a in enumerate(order):
            if a.dead is not None:
                continue
            for _ in range(PRIME):
                a.h.launch(sync=False)
            a.h.sync()
            wall, per_rank = a.h._timed_block(iters)
            errs = a.h.error_report()
            if errs:
                a.dead = "errbit:" + ";".join(errs)
                print(f"  !! {a.label} died mid-timing: {a.dead}", flush=True)
                continue
            a.samples.append((p, position, wall, max(per_rank)))
        print(f"  pass {p}: order {[a.label for a in order]}", flush=True)

    for a in arms:
        if a.dead is not None:
            continue
        problems = a.h.check_epochs() + a.h.check_signals()
        if problems:
            a.dead = "state:" + problems[0]
            print(f"  !! {a.label} state check failed: {problems[:2]}",
                  flush=True)

    out = {
        "shape": tag, "index": index, "bias": bool(bias), "iters": iters,
        "passes": passes, "reverse": REVERSE, "arms": {},
    }
    for a in arms:
        out["arms"][a.label] = {
            "plan": {key: (int(a.plan[key]) if not isinstance(a.plan[key], bool)
                           else bool(a.plan[key]))
                     for key in ("bm", "bn", "bk", "config_row", "eb",
                                 "lrow_count", "col_count", "gemm_tiles",
                                 "red_tiles", "num_reducer_ctas", "lds_bytes",
                                 "even_k", "even_n")},
            "cost": a.cost, "dead": a.dead,
            "diag": getattr(a, "diag", None),
            "max_abs_diff": a.max_abs_diff,
            "ok_graded": a.ok_graded, "ok_tight": a.ok_tight,
            "samples": [{"pass": s[0], "pos": s[1], "wall_us": s[2],
                         "device_us": s[3]} for s in a.samples],
            "best_us": min(a.walls) if a.samples else None,
            "median_us": statistics.median(a.walls) if a.samples else None,
            "spread_pct": ((max(a.walls) - min(a.walls)) / min(a.walls) * 100.0
                           if a.samples else None),
            "device_best_us": min(a.devs) if a.samples else None,
            "device_median_us": statistics.median(a.devs) if a.samples else None,
        }
    print_shape_table(out)
    for a in arms:
        a.h.close()
    del want, ref, inputs
    torch.cuda.empty_cache()
    return out


def print_shape_table(res):
    ref = res["arms"].get("T", {})
    rbest, rmed = ref.get("best_us"), ref.get("median_us")
    twin = res["arms"].get("T*", {})
    head = (f"  {'arm':>12}{'wav':>5}{'ki':>5}{'rnd':>5}"
            f"{'best':>10}{'median':>10}{'spread%':>9}"
            f"{'d_best':>10}{'d_med':>10}{'vsT_b%':>9}{'vsT_m%':>9}")
    print("  " + "-" * (len(head) - 2))
    print(head)
    for label, arm in res["arms"].items():
        c = arm["cost"]
        if arm["best_us"] is None:
            print(f"  {label:>12}{c['waves']:>5}{c['k_iters']:>5}"
                  f"{c['rounds']:>5}{'--':>10}  DEAD {arm['dead']}")
            continue
        db = (arm["best_us"] / rbest - 1) * 100 if rbest else float("nan")
        dm = (arm["median_us"] / rmed - 1) * 100 if rmed else float("nan")
        print(f"  {label:>12}{c['waves']:>5}{c['k_iters']:>5}{c['rounds']:>5}"
              f"{arm['best_us']:>10.2f}{arm['median_us']:>10.2f}"
              f"{arm['spread_pct']:>9.2f}{arm['device_best_us']:>10.2f}"
              f"{arm['device_median_us']:>10.2f}{db:>9.2f}{dm:>9.2f}")
    if rbest and twin.get("best_us"):
        nb = abs(twin["best_us"] - rbest) / min(twin["best_us"], rbest) * 100
        nm = abs(twin["median_us"] - rmed) / min(twin["median_us"], rmed) * 100
        res["null_floor_best_pct"], res["null_floor_median_pct"] = nb, nm
        print(f"  NULL ARM FLOOR (T vs T*, same tile, independent allocations): "
              f"best {nb:.2f}%  median {nm:.2f}%  <-- believe nothing below this")
    by_pos = {}
    for arm in res["arms"].values():
        if not arm["samples"]:
            continue
        med = arm["median_us"]
        for s in arm["samples"]:
            by_pos.setdefault(s["pos"], []).append(s["wall_us"] / med - 1)
    if by_pos:
        cells = " ".join(f"{p}:{statistics.mean(v) * 100:+.2f}"
                         for p, v in sorted(by_pos.items()))
        print(f"  position residual % (mean of sample/arm_median-1): {cells}")


def main():
    problems = check_derivation()
    if problems:
        print("DERIVATION SELF-CHECK FAILED -- refusing to measure:")
        for line in problems:
            print("  " + line)
        return 1
    print("derivation self-check: PASS (landed tile reproduces resolve_shape "
          "field for field on all 6 shapes)")

    rt.enable_peer_access(WORLD)
    which = (list(range(6)) if len(sys.argv) < 2 or sys.argv[1] == "all"
             else [int(x) - 1 for x in sys.argv[1].split(",")])
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    print(f"E4b tile sweep: shapes={[i + 1 for i in which]} scale={scale} "
          f"module={SWEEP_MODULE} construction_order="
          f"{'REVERSED' if REVERSE else 'forward'}")
    results = {}
    for index in which:
        iters = max(4, int(round(ITERS[index] * scale)))
        n_arms = len(ARMS[index]) + 1
        results[index] = run_shape(index, passes or n_arms, iters)

    print("\n" + "=" * 100)
    print("SUMMARY (best / median, and the null floor beside each verdict)")
    print("=" * 100)
    for index in which:
        res = results[index]
        fb = res.get("null_floor_best_pct", float("nan"))
        fm = res.get("null_floor_median_pct", float("nan"))
        base = res["arms"]["T"]
        print(f"\nshape {index + 1} {res['shape']}   null floor "
              f"best {fb:.2f}% / median {fm:.2f}%")
        for label, arm in res["arms"].items():
            if label == "T" or arm["best_us"] is None:
                continue
            gb = (base["best_us"] / arm["best_us"] - 1) * 100
            gm = (base["median_us"] / arm["median_us"] - 1) * 100
            if label == "T*":
                verdict = "(null twin)"
            elif gb > fb and gm > fm:
                verdict = "WIN vs T beyond floor -> pool and confirm"
            elif gb < -fb and gm < -fm:
                verdict = "LOSS beyond floor"
            else:
                verdict = "inside floor -> no evidence"
            print(f"  {label:<12} gain best {gb:+6.2f}%  median {gm:+6.2f}%   "
                  f"{verdict}")

    stamp = time.strftime("%H%M%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"sweep_{stamp}.json")
    with open(path, "w") as handle:
        json.dump({"shapes": {str(k): v for k, v in results.items()},
                   "passes": passes, "scale": scale},
                  handle, indent=2, default=str)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
