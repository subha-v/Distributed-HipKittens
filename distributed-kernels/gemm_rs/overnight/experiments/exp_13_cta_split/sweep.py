"""E4 re-opened: per-shape NUM_REDUCER_CTAS sweep with an explicit null arm.

Screening instrument. `num_gemm_ctas` is the only host-provided quantity the
split affects (dhk_rt.cpp:197); the device derives num_reducer_ctas, the
producer stride, tiles_per_cta and WGM from it, so a runtime override is
behaviour-identical to the same value in scored_shapes and one build serves
every arm. A winner still has to go through the full gate ladder.

Protocol, chosen because the pipelined harness has a documented positional bias
of up to 3.9% per shape:

  * every arm is a separate, independently zeroed GemmRS, and all arms for a
    shape are live at once so passes can interleave them;
  * operand tensors are generated ONCE per shape and shared by every arm, so
    arms differ only in num_gemm_ctas and in their own output heap;
  * complete rotation: passes == len(arms), order rotated by one per pass, so
    every arm occupies every position exactly once and the per-position residual
    can be reported as a direct measurement of the bias;
  * two independent arms carry the same split (the null twin). Their spread is
    the floor of this instrument;
  * best (min) and median are reported, never the mean -- per-pool jitter on
    this node attaches to an arbitrary arm.

usage: sweep.py [arms] [passes] [shapes] [iters_scale]
   e.g. sweep.py 4,8,16,24,32,40,48,56,64,80 11 all 1
        sweep.py 32,40,48 6 1,4,6 2
        sweep.py T,32 12 all 1     <- 'T' takes the split from scored_shapes,
                                      so this is a paired A/B of the landed
                                      table against uniform NR=32, and on every
                                      unchanged row the two arms are the same
                                      configuration and act as extra null arms.
"""

import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import torch                                                    # noqa: E402
import harness_lib as H                                         # noqa: E402
from harness_lib import rt, WORLD, GemmRS                       # noqa: E402

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]
# Chosen so one timed block is ~25-45 ms of device work on every shape: short
# blocks measure block edges, long ones waste the night.
ITERS = [300, 250, 250, 120, 50, 25]
PRIME = 4                     # unsynced launches to fill the pipe before timing
NULL_TWIN = 32                # the split that is measured twice
TIGHT = 2e-3
GRADED = 1e-2


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


class Arm:
    def __init__(self, label, nr, handle):
        self.label = label
        self.nr = nr
        self.h = handle
        self.dead = None          # reason string once it fails
        self.samples = []         # (pass, position, wall_us, device_us)

    @property
    def walls(self):
        return [s[2] for s in self.samples]

    @property
    def devs(self):
        return [s[3] for s in self.samples]


def geometry(plan, nr):
    """Producer waves / reducer rounds / WGM, all derivable before running."""
    tiles = int(plan["gemm_tiles"])
    ng = 304 - nr
    num_pid_m = int(plan["m"]) // int(plan["bm"])
    return {
        "ng": ng,
        "waves": (tiles + ng - 1) // ng,
        "rounds": (int(plan["red_tiles"]) + nr - 1) // nr,
        "wgm": 4 if tiles <= ng else num_pid_m,
    }


def arm_specs(splits):
    specs = [("T" if nr is None else str(nr), nr) for nr in splits]
    if NULL_TWIN in splits:
        # The twin is inserted at the far end of the list so the two copies are
        # maximally separated in every rotation.
        specs.append((f"{NULL_TWIN}*", NULL_TWIN))
    return specs


def run_shape(index, shape, splits, passes, iters):
    m, n, k, bias, seed = shape
    tag = f"{m}x{n}x{k}"
    inputs = [H.generate_input(r, WORLD, m, n, k, bias, seed)
              for r in range(WORLD)]
    ref = H.reference_reduce_scatter(inputs, m, n)
    want = [ref[r].to(f"cuda:{r}") for r in range(WORLD)]

    arms, plan = [], None
    for label, nr in arm_specs(splits):
        h = GemmRS(m, n, k, bias, num_reducer_ctas=nr)
        h.inputs = inputs                  # shared operands, identical addresses
        h._rebuild_args()
        plan = h.plan
        # nr is None for the 'T' arm; the resolved value comes back from the plan
        # so the geometry columns are right for both kinds of arm.
        arms.append(Arm(label, int(h.plan["num_reducer_ctas"]), h))

    geo = {a.label: geometry(plan, a.nr) for a in arms}
    print(f"\n===== shape {index + 1}: {tag} bias={int(bias)}  "
          f"row={plan['config_row']} {plan['bm']}/{plan['bn']}/{plan['bk']}  "
          f"gemm_tiles={plan['gemm_tiles']} red_tiles={plan['red_tiles']}  "
          f"iters={iters} passes={passes} arms={len(arms)} =====", flush=True)

    # Correctness first, at both tolerances, on every arm. A starved split is a
    # protocol stress test; an arm that fails here is never timed.
    for a in arms:
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
        a.max_abs_diff = worst
        a.ok_graded, a.ok_tight = ok_graded, ok_tight
        if errs:
            a.dead = "errbit:" + ";".join(errs)
        elif not ok_graded:
            a.dead = f"WRONG@1e-2 max|diff|={worst:.2e}"
        elif not ok_tight:
            a.dead = f"WRONG@2e-3 max|diff|={worst:.2e}"
        print(f"  NR={a.label:<4}({a.nr:>3}) NG={geo[a.label]['ng']:<4} "
              f"waves={geo[a.label]['waves']} rounds={geo[a.label]['rounds']:<3} "
              f"WGM={geo[a.label]['wgm']:<3} max|diff|={worst:.2e} "
              f"{'OK' if a.dead is None else 'DEAD ' + a.dead}", flush=True)

    live = [a for a in arms if a.dead is None]
    # Global warmup cycling every arm, duration based: idle sclk here is ~132
    # MHz against ~1900 loaded even with perf determinism pinned.
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
                print(f"  !! NR={a.label} died mid-timing: {a.dead}", flush=True)
                continue
            a.samples.append((p, position, wall, max(per_rank)))
        print(f"  pass {p}: order {[a.label for a in order]}", flush=True)

    # Protocol bookkeeping after the fact: every scheduled CTA's epoch cell and
    # every touched ready/credit cell must read the exact launch count.
    for a in arms:
        if a.dead is not None:
            continue
        problems = a.h.check_epochs() + a.h.check_signals()
        if problems:
            a.dead = "state:" + problems[0]
            print(f"  !! NR={a.label} state check failed: {problems[:2]}",
                  flush=True)

    out = {
        "shape": tag, "index": index, "bias": bool(bias), "iters": iters,
        "passes": passes,
        "plan": {key: plan[key] for key in
                 ("config_row", "bm", "bn", "bk", "eb", "lrow_count",
                  "col_count", "gemm_tiles", "red_tiles")},
        "arms": {},
    }
    for a in arms:
        out["arms"][a.label] = {
            "nr": a.nr, "dead": a.dead, "geometry": geo[a.label],
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
    ref = res["arms"].get(str(NULL_TWIN), {})
    rbest, rmed = ref.get("best_us"), ref.get("median_us")
    twin = res["arms"].get(f"{NULL_TWIN}*", {})
    head = (f"  {'NR':>5}{'NG':>5}{'wav':>5}{'rnd':>5}{'WGM':>5}"
            f"{'best':>10}{'median':>10}{'spread%':>9}"
            f"{'d_best':>10}{'d_med':>10}{'vs32_b%':>9}{'vs32_m%':>9}")
    print("  " + "-" * (len(head) - 2))
    print(head)
    for label, arm in res["arms"].items():
        g = arm["geometry"]
        if arm["best_us"] is None:
            print(f"  {label:>5}{g['ng']:>5}{g['waves']:>5}{g['rounds']:>5}"
                  f"{g['wgm']:>5}{'--':>10}  DEAD {arm['dead']}")
            continue
        db = (arm["best_us"] / rbest - 1) * 100 if rbest else float("nan")
        dm = (arm["median_us"] / rmed - 1) * 100 if rmed else float("nan")
        print(f"  {label:>5}{g['ng']:>5}{g['waves']:>5}{g['rounds']:>5}"
              f"{g['wgm']:>5}{arm['best_us']:>10.2f}{arm['median_us']:>10.2f}"
              f"{arm['spread_pct']:>9.2f}{arm['device_best_us']:>10.2f}"
              f"{arm['device_median_us']:>10.2f}{db:>9.2f}{dm:>9.2f}")
    if rbest and twin.get("best_us"):
        nb = abs(twin["best_us"] - rbest) / min(twin["best_us"], rbest) * 100
        nm = abs(twin["median_us"] - rmed) / min(twin["median_us"], rmed) * 100
        res["null_floor_best_pct"], res["null_floor_median_pct"] = nb, nm
        print(f"  NULL ARM FLOOR (NR={NULL_TWIN} vs its twin): "
              f"best {nb:.2f}%  median {nm:.2f}%   <-- believe nothing below this")
    # Position residual: how much a slot in the sweep order is worth, measured.
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
        print(f"  position residual %% (mean of sample/arm_median-1): {cells}")


def main():
    rt.enable_peer_access(WORLD)
    splits = [None if x == "T" else int(x) for x in
              (sys.argv[1] if len(sys.argv) > 1
               else "4,8,16,24,32,40,48,56,64,80").split(",")]
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    which = (list(range(6)) if len(sys.argv) < 4 or sys.argv[3] == "all"
             else [int(x) - 1 for x in sys.argv[3].split(",")])
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    n_arms = len(arm_specs(splits))
    if passes <= 0:
        passes = n_arms                      # complete positional rotation

    print(f"E4 re-open: splits={splits} twin={NULL_TWIN} arms={n_arms} "
          f"passes={passes} shapes={[i + 1 for i in which]} scale={scale}")
    results = {}
    for index in which:
        iters = max(4, int(round(ITERS[index] * scale)))
        results[index] = run_shape(index, SCORED[index], splits, passes, iters)

    print("\n" + "=" * 112)
    print("UNIFORM-NR CURVE: per-shape best (and median), geomean over the "
          "measured shapes")
    print("=" * 112)
    labels = [lb for lb, _ in arm_specs(splits)]
    head = f"{'NR':>6}" + "".join(f"{f'shape{i + 1}':>12}" for i in which) + \
           f"{'geomean':>11}{'vs NR=32':>10}"
    print(head)
    print("-" * len(head))
    curve = {}
    for stat in ("best_us", "median_us"):
        print(f"  [{stat}]")
        base = None
        for label in labels:
            cells, vals, ok = "", [], True
            for index in which:
                arm = results[index]["arms"][label]
                if arm[stat] is None:
                    cells += f"{'DEAD':>12}"
                    ok = False
                else:
                    cells += f"{arm[stat]:>12.2f}"
                    vals.append(arm[stat])
            gm = geomean(vals) if ok and vals else float("nan")
            if label == str(NULL_TWIN):
                base = gm
            rel = (gm / base - 1) * 100 if base else float("nan")
            curve.setdefault(stat, {})[label] = gm
            print(f"{label:>6}{cells}{gm:>11.2f}{rel:>9.2f}%")

    print("\nPER-SHAPE ARGMIN (screening only, and only trust a pick whose "
          "margin exceeds that shape's null floor)")
    picks = {}
    for index in which:
        res = results[index]
        floor_b = res.get("null_floor_best_pct", float("nan"))
        floor_m = res.get("null_floor_median_pct", float("nan"))
        live = {lb: a for lb, a in res["arms"].items()
                if a["best_us"] is not None and not lb.endswith("*")}
        pb = min(live, key=lambda lb: live[lb]["best_us"])
        pm = min(live, key=lambda lb: live[lb]["median_us"])
        base = res["arms"][str(NULL_TWIN)]
        gain_b = (base["best_us"] / live[pb]["best_us"] - 1) * 100
        gain_m = (base["median_us"] / live[pm]["median_us"] - 1) * 100
        picks[index] = {"argmin_best": pb, "gain_best_pct": gain_b,
                        "argmin_median": pm, "gain_median_pct": gain_m,
                        "null_floor_best_pct": floor_b,
                        "null_floor_median_pct": floor_m}
        verdict = ("MOVED" if gain_b > floor_b and gain_m > floor_m
                   else "inside floor -> keep 32")
        print(f"  shape {index + 1} {res['shape']:<20} best NR={pb:<4} "
              f"(+{gain_b:5.2f}% vs 32)   median NR={pm:<4} "
              f"(+{gain_m:5.2f}%)   null floor {floor_b:.2f}/{floor_m:.2f}%  "
              f"{verdict}")

    stamp = time.strftime("%H%M%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"sweep_{stamp}.json")
    with open(path, "w") as handle:
        json.dump({"splits": splits, "passes": passes, "scale": scale,
                   "twin": NULL_TWIN, "curve": curve, "picks": picks,
                   "shapes": {str(k): v for k, v in results.items()}},
                  handle, indent=2, default=str)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
