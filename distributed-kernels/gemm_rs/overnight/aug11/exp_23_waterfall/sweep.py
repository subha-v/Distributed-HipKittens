"""exp_23: the knob waterfall, same-run paired over all six graded shapes.

Eight arms per shape, all alive at once, all reading the SAME operand tensors
at the same addresses, differing only in which rung binary they load and how
many reducer CTAs they are given:

    a     WGM4=1 RELEASE_GROUP=1   the aug10 baseline
    b     WGM4=0 RELEASE_GROUP=1   + WGM destination-spreading task order
    c     WGM4=0 RELEASE_GROUP=4   + grouped release (the shipped best)
    null  identical to c, different module name  <- the allocation-noise floor
    nr8 nr16 nr32 nr48   rung c, num_reducer_ctas overridden (the Q2 exhibit)

Three things about the instrument, all of them load-bearing.

ONE. The dominant error term on this node is a per-allocation offset, not
positional drift: two arms with identical code and config have measured 4.28%
apart on shape 6 while the per-position residual stayed inside 0.47%. So `null`
is not decoration -- it is the only thing that makes an error bar on this plot
mean anything, and every delta is reported beside it. Published floors
(1.34/0.56/0.61/2.41/2.17/4.28 %) are LOWER bounds; a re-measurement came back
2-2.6x worse on shapes 1-3.

TWO. Part of that bias is allocation ORDER, not chance. `SWEEP_REVERSE=1`
reverses the order in which arms are CONSTRUCTED (not the timing rotation), so
running half the draws reversed makes the pooled null set symmetric instead of
systematically signed.

THREE. Means are unusable here. Best (min) and median only, and the geometric
mean over the six shapes is the ranking statistic.

No source is edited by this experiment: every rung is a -D flag revert built
into this directory's own build/, and harness_lib is pointed at it rather than
at harness/build (where another agent's ablation arms live).

usage:
    sweep.py [shapes] [passes] [iters_scale]
        sweep.py all               <- all six, complete 8-arm rotation
        sweep.py 5,6 8 1
    SWEEP_REVERSE=1 sweep.py all   <- reversed construction order
    sweep.py --pool                <- rebuild waterfall.json from draw_*.json
"""

import glob
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "harness"))

import torch                                                     # noqa: E402
import harness_lib as H                                          # noqa: E402
from harness_lib import rt, WORLD, GemmRS                        # noqa: E402

# dhk_rt has already been imported from harness/build by the line above, which
# is what we want -- it is shared, unchanged and read-only. Every KERNEL module
# from here on comes out of this experiment's own build directory.
H.BUILD_DIR = os.path.join(HERE, "build")

MOD = {"a": "gemm_rs_w23_a", "b": "gemm_rs_w23_b",
       "c": "gemm_rs_w23_c", "null": "gemm_rs_w23_null"}
RUNG_CONFIG = {
    "a": {"WGM4": 1, "RELEASE_GROUP": 1, "RELEASE_GROUP_FULL_ONLY": 1},
    "b": {"WGM4": 0, "RELEASE_GROUP": 1, "RELEASE_GROUP_FULL_ONLY": 1},
    "c": {"WGM4": 0, "RELEASE_GROUP": 4, "RELEASE_GROUP_FULL_ONLY": 1},
    "null": {"WGM4": 0, "RELEASE_GROUP": 4, "RELEASE_GROUP_FULL_ONLY": 1},
}
NR_POINTS = [8, 16, 32, 48]
RUNG_ORDER = ["a", "b", "c", "null"]

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]
# One timed block is ~50-100 ms of device work on every shape, and never fewer
# than the charter's 50 iterations on the two large ones.
ITERS = [300, 250, 250, 120, 60, 50]
PRIME = 4
WARMUP_MS = 400
TIGHT = 2e-3
GRADED = 1e-2

REVERSE = os.environ.get("SWEEP_REVERSE") == "1"


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def arm_specs():
    """(label, module, num_reducer_ctas) for one shape. NR None = shipped."""
    specs = [(lab, MOD[lab], None) for lab in RUNG_ORDER]
    specs += [(f"nr{n}", MOD["c"], n) for n in NR_POINTS]
    return specs


class Arm:
    def __init__(self, label, module, nr, handle):
        self.label = label
        self.module = module
        self.nr = nr
        self.h = handle
        self.dead = None
        self.samples = []           # (pass, position, wall_us, device_us)

    @property
    def walls(self):
        return [s[2] for s in self.samples]

    @property
    def devs(self):
        return [s[3] for s in self.samples]


def diagnose_dead(arm, want):
    """Separate "ran and computed the wrong thing" from "never ran".

    Both look identical through a tolerance check -- an output left at its
    initial zeros reports max|diff| == max|ref|. The epoch cell discriminates:
    every scheduled CTA increments it at entry, before any protocol step, so
    cells still at 0 after a launch prove the launch itself did not happen.
    """
    zero_ranks = sum(1 for r in range(WORLD) if not bool(arm.h.out[r].any()))
    cells = arm.h.epoch_cells(0)
    prod = cells[:int(arm.h.plan["num_gemm_ctas"])]
    red = cells[rt.EP_RED_U32:
                rt.EP_RED_U32 + int(arm.h.plan["num_reducer_ctas"])]
    ref_absmax = max(float(want[r].float().abs().max()) for r in range(WORLD))
    note = (f"    diag: out all-zero on {zero_ranks}/{WORLD} ranks; rank0 "
            f"epoch producer max={max(prod)} reducer max={max(red)} "
            f"(expect {arm.h.n_calls}); max|ref|={ref_absmax:.2e}")
    if zero_ranks == WORLD and max(prod) == 0:
        note += "\n    => THE LAUNCH NEVER RAN (not a numerics bug)"
    elif max(prod) == arm.h.n_calls:
        note += "\n    => kernel ran to completion and computed wrong values"
    return note


def run_shape(index, passes, iters):
    m, n, k, bias, seed = SCORED[index]
    tag = f"{m}x{n}x{k}"
    inputs = [H.generate_input(r, WORLD, m, n, k, bias, seed)
              for r in range(WORLD)]
    ref = H.reference_reduce_scatter(inputs, m, n)
    want = [ref[r].to(f"cuda:{r}") for r in range(WORLD)]

    specs = arm_specs()
    if REVERSE:
        specs = list(reversed(specs))

    arms = []
    for label, module, nr in specs:
        handle = GemmRS(m, n, k, bias, num_reducer_ctas=nr, module_name=module)
        handle.inputs = inputs            # shared operands, same addresses
        handle._rebuild_args()
        arms.append(Arm(label, module, nr, handle))
    arms_by_label = {a.label: a for a in arms}

    shipped_nr = int(arms_by_label["c"].h.plan["num_reducer_ctas"])
    plan = arms_by_label["c"].h.plan
    tiles = (m // int(plan["bm"])) * int(plan["col_count"])
    ng = int(plan["num_gemm_ctas"])
    tiles_per_cta = -(-tiles // ng)
    print(f"\n===== shape {index + 1}: {tag} bias={int(bias)}  "
          f"{plan['bm']}/{plan['bn']}/{plan['bk']} row={plan['config_row']} "
          f"shippedNR={shipped_nr} NG={ng} tiles={tiles} "
          f"tiles/CTA={tiles_per_cta}  iters={iters} passes={passes} "
          f"arms={len(arms)} construction="
          f"{'REVERSED' if REVERSE else 'forward'} =====", flush=True)
    # The two rungs are conditionally active and the conditions are known before
    # any GPU runs (plan.md pre-registers this). Printed so the reader can tell
    # a flat rung that SHOULD be flat from one that failed.
    print(f"  a->b active on this shape: {tiles > ng}   "
          f"b->c active on this shape: {tiles_per_cta >= 4}", flush=True)

    for a in arms:
        a.h.launch()
        errs = a.h.error_report()
        worst, ok_tight, ok_graded = 0.0, True, True
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
            # Tolerances are gates, not dials: clearing only the graded one is a
            # regression, so the arm is not plotted.
            a.dead = f"WRONG@2e-3 max|diff|={worst:.2e}"
        print(f"  {a.label:<6} mod={a.module:<18} "
              f"NR={int(a.h.plan['num_reducer_ctas']):<3} "
              f"NG={int(a.h.plan['num_gemm_ctas']):<4} "
              f"max|diff|={worst:.2e} "
              f"{'OK' if a.dead is None else 'DEAD ' + a.dead}", flush=True)
        if a.dead is not None:
            a.diag = diagnose_dead(a, want)
            print(a.diag, flush=True)

    live = [a for a in arms if a.dead is None]
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < WARMUP_MS and live:
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
        "passes": passes, "reverse": REVERSE, "shipped_nr": shipped_nr,
        "tiles": tiles, "num_gemm_ctas": ng, "tiles_per_cta": tiles_per_cta,
        "ab_rung_active": bool(tiles > ng),
        "bc_rung_active": bool(tiles_per_cta >= 4),
        "arms": {},
    }
    for a in arms:
        out["arms"][a.label] = {
            "module": a.module,
            "nr": int(a.h.plan["num_reducer_ctas"]),
            "num_gemm_ctas": int(a.h.plan["num_gemm_ctas"]),
            "dead": a.dead, "diag": getattr(a, "diag", None),
            "max_abs_diff": a.max_abs_diff,
            "ok_graded": a.ok_graded, "ok_tight": a.ok_tight,
            "samples": [{"pass": s[0], "pos": s[1], "wall_us": s[2],
                         "device_us": s[3]} for s in a.samples],
            "best_us": min(a.walls) if a.samples else None,
            "median_us": statistics.median(a.walls) if a.samples else None,
            "spread_pct": ((max(a.walls) - min(a.walls)) / min(a.walls) * 100.0
                           if a.samples else None),
            "device_best_us": min(a.devs) if a.samples else None,
            "device_median_us": (statistics.median(a.devs) if a.samples
                                 else None),
        }
    print_shape_table(out)
    for a in arms:
        a.h.close()
    del want, ref, inputs
    torch.cuda.empty_cache()
    return out


def print_shape_table(res):
    arms = res["arms"]
    base = arms.get("a", {})
    ab, am = base.get("best_us"), base.get("median_us")
    head = (f"  {'arm':>7}{'NR':>5}{'best':>10}{'median':>10}{'spread%':>9}"
            f"{'d_best':>10}{'vs_a_b%':>9}{'vs_a_m%':>9}")
    print("  " + "-" * (len(head) - 2))
    print(head)
    for label, arm in arms.items():
        if arm["best_us"] is None:
            print(f"  {label:>7}{arm['nr']:>5}{'--':>10}  DEAD {arm['dead']}")
            continue
        db = (arm["best_us"] / ab - 1) * 100 if ab else float("nan")
        dm = (arm["median_us"] / am - 1) * 100 if am else float("nan")
        print(f"  {label:>7}{arm['nr']:>5}{arm['best_us']:>10.2f}"
              f"{arm['median_us']:>10.2f}{arm['spread_pct']:>9.2f}"
              f"{arm['device_best_us']:>10.2f}{db:>9.2f}{dm:>9.2f}")
    c, nul = arms.get("c", {}), arms.get("null", {})
    if c.get("best_us") and nul.get("best_us"):
        fb = abs(c["best_us"] - nul["best_us"]) / min(c["best_us"],
                                                      nul["best_us"]) * 100
        fm = abs(c["median_us"] - nul["median_us"]) / min(
            c["median_us"], nul["median_us"]) * 100
        res["null_floor_best_pct"], res["null_floor_median_pct"] = fb, fm
        print(f"  NULL FLOOR (c vs null, identical device ISA, independent "
              f"allocations): best {fb:.2f}%  median {fm:.2f}%  "
              f"<-- believe nothing below this")
    by_pos = {}
    for arm in arms.values():
        if not arm["samples"]:
            continue
        med = arm["median_us"]
        for s in arm["samples"]:
            by_pos.setdefault(s["pos"], []).append(s["wall_us"] / med - 1)
    if by_pos:
        cells = " ".join(f"{p}:{statistics.mean(v) * 100:+.2f}"
                         for p, v in sorted(by_pos.items()))
        print(f"  position residual %: {cells}")


# ---------------------------------------------------------------------------
# waterfall.json -- the committed plot-ready schema, pooled over every draw
# present in this directory. Pooling happens here rather than in a separate
# tool because the unit of evidence is one DRAW (one process = one hipMalloc
# outcome), so a single draw's ratios are not the deliverable.
# ---------------------------------------------------------------------------
def fingerprint_shas():
    path = os.path.join(HERE, "fingerprints.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    return {lab: e.get("isa_sha256") for lab, e in data.get("rungs", {}).items()}


def pool(paths):
    draws = []
    for path in sorted(paths):
        draws.append((os.path.basename(path), json.load(open(path))))
    if not draws:
        return None
    shas = fingerprint_shas()

    indices = sorted({int(key) for _, d in draws for key in d["shapes"]})
    # samples[label][index] = every wall sample from every draw
    samples, meta = {}, {}
    for _, d in draws:
        for key, res in d["shapes"].items():
            idx = int(key)
            meta[idx] = {"shape": res["shape"],
                         "ab_rung_active": res.get("ab_rung_active"),
                         "bc_rung_active": res.get("bc_rung_active"),
                         "shipped_nr": res.get("shipped_nr")}
            for label, arm in res["arms"].items():
                if arm["best_us"] is None:
                    continue
                samples.setdefault(label, {}).setdefault(idx, []).extend(
                    s["wall_us"] for s in arm["samples"])

    def best(label, idx):
        vals = samples.get(label, {}).get(idx)
        return min(vals) if vals else None

    def med(label, idx):
        vals = samples.get(label, {}).get(idx)
        return statistics.median(vals) if vals else None

    # The floor is the WORST single draw of |c - null|, because that is the
    # number a delta has to clear to be believable at all. A floor averaged
    # over draws would flatter every rung.
    floors = {}
    for idx in indices:
        per_draw = []
        for _, d in draws:
            res = d["shapes"].get(str(idx))
            if not res:
                continue
            c, nul = res["arms"].get("c"), res["arms"].get("null")
            if not c or not nul or c["median_us"] is None or \
                    nul["median_us"] is None:
                continue
            per_draw.append(abs(c["median_us"] - nul["median_us"]) /
                            min(c["median_us"], nul["median_us"]) * 100)
        floors[idx] = max(per_draw) if per_draw else None

    prev = {"a": None, "b": "a", "c": "b", "null": "c"}
    rungs = []
    for label in RUNG_ORDER:
        per_shape = []
        for idx in indices:
            b, mv = best(label, idx), med(label, idx)
            if mv is None:
                continue
            base_a = med("a", idx)
            base_p = med(prev[label], idx) if prev[label] else None
            per_shape.append({
                "shape": meta[idx]["shape"],
                "shape_index": idx + 1,
                "best_us": b,
                "median_us": mv,
                "samples_us": samples[label][idx],
                "null_floor_pct": floors[idx],
                "ratio_vs_prev_rung": (mv / base_p) if base_p else 1.0,
                "ratio_vs_rung_a": (mv / base_a) if base_a else None,
                "rung_active_on_this_shape": (
                    meta[idx]["ab_rung_active"] if label == "b" else
                    meta[idx]["bc_rung_active"] if label == "c" else None),
            })
        if not per_shape:
            continue
        rungs.append({
            "rung": label,
            "config": RUNG_CONFIG[label],
            "fingerprint_sha": shas.get(label),
            "per_shape": per_shape,
            "geomean_best_us": geomean([p["best_us"] for p in per_shape]),
            "geomean_median_us": geomean([p["median_us"] for p in per_shape]),
        })

    nr_sweep = []
    for nr in NR_POINTS:
        label = f"nr{nr}"
        per_shape = []
        for idx in indices:
            b, mv = best(label, idx), med(label, idx)
            if mv is None:
                continue
            base_c = med("c", idx)
            per_shape.append({
                "shape": meta[idx]["shape"],
                "shape_index": idx + 1,
                "best_us": b,
                "median_us": mv,
                "samples_us": samples[label][idx],
                "null_floor_pct": floors[idx],
                "ratio_vs_shipped_nr": (mv / base_c) if base_c else None,
                "shipped_nr": meta[idx]["shipped_nr"],
            })
        if not per_shape:
            continue
        nr_sweep.append({
            "nr": nr,
            "config": RUNG_CONFIG["c"],
            "fingerprint_sha": shas.get("c"),
            "per_shape": per_shape,
            "geomean_best_us": geomean([p["best_us"] for p in per_shape]),
            "geomean_median_us": geomean([p["median_us"] for p in per_shape]),
        })

    return {
        "experiment": "exp_23_waterfall",
        "protocol": "pipelined, same-run paired, 8 arms per shape, complete "
                    "rotation, best+median only (means are unusable on this "
                    "node)",
        "draws": [name for name, _ in draws],
        "draws_reversed": sum(1 for _, d in draws if d.get("reverse")),
        "rungs": rungs,
        "nr_sweep": nr_sweep,
    }


def write_waterfall():
    paths = sorted(glob.glob(os.path.join(HERE, "draw_*.json")))
    data = pool(paths)
    if data is None:
        print("no draw_*.json to pool")
        return 1
    path = os.path.join(HERE, "waterfall.json")
    json.dump(data, open(path, "w"), indent=2)
    print(f"\npooled {len(paths)} draw(s) "
          f"({data['draws_reversed']} reversed) -> {path}")
    print(f"\n{'rung':>6}{'geo_best':>11}{'geo_median':>12}"
          f"{'vs_a':>9}{'vs_prev':>9}")
    prev = None
    base = data["rungs"][0]["geomean_median_us"] if data["rungs"] else None
    for rung in data["rungs"]:
        gm = rung["geomean_median_us"]
        print(f"{rung['rung']:>6}{rung['geomean_best_us']:>11.2f}{gm:>12.2f}"
              f"{gm / base:>9.4f}"
              f"{(gm / prev if prev else 1.0):>9.4f}")
        prev = gm
    if data["nr_sweep"]:
        print(f"\n{'NR':>6}{'geo_best':>11}{'geo_median':>12}{'vs_c':>9}")
        cbase = next((r["geomean_median_us"] for r in data["rungs"]
                      if r["rung"] == "c"), None)
        for entry in data["nr_sweep"]:
            gm = entry["geomean_median_us"]
            print(f"{entry['nr']:>6}{entry['geomean_best_us']:>11.2f}"
                  f"{gm:>12.2f}{(gm / cbase if cbase else float('nan')):>9.4f}")
    return 0


def main():
    if "--pool" in sys.argv:
        return write_waterfall()

    fp = os.path.join(HERE, "fingerprints.json")
    if not os.path.exists(fp):
        print("fingerprints.json missing -- run fingerprint.py first. Without "
              "it a stale build can masquerade as a rung and this whole figure "
              "is four measurements of one binary.")
        return 1
    data = json.load(open(fp))
    hard = [a for a in data.get("assertions", [])
            if a.get("hard") and not a.get("passed")]
    if hard:
        print("fingerprint gate FAILED -- refusing to measure:")
        for a in hard:
            print(f"  {a['id']}: {a['detail']}")
        return 1
    shas = {lab: e["isa_sha256"] for lab, e in data["rungs"].items()}
    print("fingerprint gate: PASS")
    for lab in RUNG_ORDER:
        print(f"  rung {lab:<5} isa {shas[lab][:16]} "
              f"config {RUNG_CONFIG[lab]}")

    rt.enable_peer_access(WORLD)
    which = (list(range(6)) if len(sys.argv) < 2 or sys.argv[1].startswith("-")
             or sys.argv[1] == "all"
             else [int(x) - 1 for x in sys.argv[1].split(",")])
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    n_arms = len(arm_specs())

    print(f"\nexp_23 waterfall: shapes={[i + 1 for i in which]} scale={scale} "
          f"construction={'REVERSED' if REVERSE else 'forward'}")
    results = {}
    for index in which:
        iters = max(50, int(round(ITERS[index] * scale)))
        results[index] = run_shape(index, passes or n_arms, iters)

    print("\n" + "=" * 100)
    print("DRAW SUMMARY (median us; every delta beside its null floor)")
    print("=" * 100)
    for index in which:
        res = results[index]
        floor = res.get("null_floor_median_pct", float("nan"))
        base = res["arms"]["a"]
        print(f"\nshape {index + 1} {res['shape']}  null floor "
              f"{floor:.2f}%  a->b active={res['ab_rung_active']}  "
              f"b->c active={res['bc_rung_active']}")
        prev_label = {"b": "a", "c": "b", "null": "c"}
        for label in ("b", "c", "null"):
            arm = res["arms"][label]
            if arm["median_us"] is None or base["median_us"] is None:
                continue
            gp = (res["arms"][prev_label[label]]["median_us"] /
                  arm["median_us"] - 1) * 100
            ga = (base["median_us"] / arm["median_us"] - 1) * 100
            verdict = ("beyond floor" if abs(gp) > floor
                       else "inside floor -> no evidence")
            print(f"  {label:<6} gain vs {prev_label[label]:<4} {gp:+7.2f}%   "
                  f"vs a {ga:+7.2f}%   {verdict}")
        cm = res["arms"]["c"]["median_us"]
        for nr in NR_POINTS:
            arm = res["arms"].get(f"nr{nr}", {})
            if not arm.get("median_us") or not cm:
                continue
            g = (cm / arm["median_us"] - 1) * 100
            print(f"  NR={nr:<4} vs shipped NR {g:+7.2f}%   "
                  f"{'beyond floor' if abs(g) > floor else 'inside floor'}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tagged = "rev" if REVERSE else "fwd"
    path = os.path.join(HERE, f"draw_{stamp}_{tagged}.json")
    json.dump({"shapes": {str(k): v for k, v in results.items()},
               "passes": passes, "scale": scale, "reverse": REVERSE,
               "fingerprints": shas},
              open(path, "w"), indent=2, default=str)
    print(f"\nwrote {path}")
    write_waterfall()
    return 0


if __name__ == "__main__":
    sys.exit(main())
