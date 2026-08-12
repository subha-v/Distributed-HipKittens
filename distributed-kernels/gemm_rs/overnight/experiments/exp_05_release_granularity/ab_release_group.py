"""exp_05 E3: the RELEASE_GROUP arms, paired inside one process.

Why this exists. The gate ladder measures one arm per run, and this node's
run-to-run spread on an identical binary is ~1.3% -- larger than the effect on
four of the six graded shapes. Cross-run per-shape numbers therefore cannot
decide between RELEASE_GROUP = 1, 2 and 4. Here every arm is a separately
compiled module, all of them are open at once, and their timed blocks are
INTERLEAVED in a rotating order inside one round: adjacent samples share the
clock state, the allocation state and the input sequence, so the only difference
between them is the release granularity.

Per shape it also asserts that the arms are BITWISE identical to each other.
That is a stronger form of the m9 golden check and it is free here, because all
the arms are driven over the same inputs in the same process: E3 must not
perturb a single bit of arithmetic, only when the release fence is issued.

The last arm, `rg1b`, is a NULL arm: a second build of RELEASE_GROUP = 1 under a
different module name. Nothing an instruction can see separates it from rg1, so
whatever the A/B reports between them is the instrument's own bias -- allocation
order, arm order within a round, which heap each arm happened to get -- and it
is the floor under which no other arm's delta means anything. The first pass of
this experiment claimed a 2% regression on a shape with ONE tile per producer
CTA, which cannot group at all, and this arm exists to price that claim.

Reports best and median per arm, never the mean: with a 1.3% floor a mean over
few samples is not a usable statistic, and the ledger's convention is best and
median.

  python3 ab_release_group.py [rounds] [iters] [warmup_ms] [reverse]
"""

import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import harness_lib as H                                   # noqa: E402
from harness_lib import rt, WORLD, GemmRS                 # noqa: E402

# (tag, RELEASE_GROUP). rg1 is the behaviour-preserving control arm, rg4c groups
# only where a CTA owns a full group, and rg1b is the null arm -- the same
# constant as rg1, a second module, so its delta is pure instrument.
ARMS = [("rg1", 1), ("rg2", 2), ("rg4", 4), ("rg4c", 4), ("rg1b", 1)]
NULL_PAIR = ("rg1", "rg1b")

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

# Tiles per producer CTA decides whether an arm can do anything at all: a shape
# with one tile per CTA has emitted == 1 for every group and every arm collapses
# to the same instruction stream. Those shapes are the built-in controls.
TIGHT = 2e-3


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def per_cta(plan):
    tiles = int(plan["gemm_tiles"])
    producers = int(plan["num_gemm_ctas"])
    return tiles, producers, -(-tiles // producers)


def measure_shape(shape, rounds, iters, warmup_ms):
    m, n, k, bias, seed = shape
    arms = {}
    try:
        for tag, _ in ARMS:
            h = GemmRS(m, n, k, bias, module_name=f"gemm_rs_mi300x_{tag}")
            h.set_inputs(seed)
            arms[tag] = h
        plan = arms[ARMS[0][0]].plan
        tiles, producers, ppc = per_cta(plan)
        print(f"\n=== m={m} n={n} k={k} bias={int(bias)} "
              f"(row {int(plan['config_row'])}) ===")
        print(f"    {tiles} tiles over {producers} producer CTAs "
              f"= up to {ppc} tiles per CTA"
              f"{'  <-- built-in control, no arm can group' if ppc == 1 else ''}")

        # Correctness and cross-arm bit-exactness before any timing.
        ref = None
        for tag, _ in ARMS:
            h = arms[tag]
            h.launch()
            checks = h.verify(rtol=TIGHT, atol=TIGHT)
            assert all(c["allclose"] for c in checks), f"{tag} failed {TIGHT}"
            assert not h.error_report(), f"{tag} raised {h.error_report()}"
            if ref is None:
                ref = [h.out[r].clone() for r in range(WORLD)]
            else:
                same = all(torch.equal(h.out[r], ref[r]) for r in range(WORLD))
                assert same, f"{tag} is NOT bit-identical to {ARMS[0][0]}"
        print(f"    all {len(ARMS)} arms correct at {TIGHT} and bit-identical "
              f"to each other on all 8 ranks")

        # Shared duration-based warmup: idle sclk here is ~132 MHz against
        # ~1900 under load, so an unwarmed block measures a ramping GPU.
        started = time.perf_counter()
        while (time.perf_counter() - started) * 1e3 < warmup_ms:
            for tag, _ in ARMS:
                arms[tag].launch(sync=False)
            for tag, _ in ARMS:
                arms[tag].sync()

        samples = {tag: [] for tag, _ in ARMS}
        for r in range(rounds):
            order = [(r + i) % len(ARMS) for i in range(len(ARMS))]
            for index in order:
                tag = ARMS[index][0]
                wall, _ = arms[tag]._timed_block(iters)
                samples[tag].append(wall)

        out = {}
        base = NULL_PAIR[0]
        for tag, group in ARMS:
            values = samples[tag]
            out[tag] = {
                "release_group": group,
                "best": min(values),
                "median": statistics.median(values),
                "spread_pct": (max(values) - min(values)) / min(values) * 100.0,
                "samples": values,
            }
        for tag, _ in ARMS:
            row = out[tag]
            print(f"    {tag}: best {row['best']:9.2f}  "
                  f"median {row['median']:9.2f}  "
                  f"spread {row['spread_pct']:5.2f}%  "
                  f"best x{row['best'] / out[base]['best']:.4f}  "
                  f"median x{row['median'] / out[base]['median']:.4f}")
        for tag, _ in ARMS:
            assert not arms[tag].error_report(), f"{tag} raised late"
        return out
    finally:
        for h in arms.values():
            h.close()


def main():
    global ARMS
    rt.enable_peer_access(WORLD)
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    warmup_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    if len(sys.argv) > 4 and sys.argv[4] not in ("0", "", "no"):
        # Reversing the arm list reverses allocation order too, which is the
        # other half of the null arm's job: a bias that survives both
        # directions is not a bias.
        ARMS = list(reversed(ARMS))

    print("=" * 96)
    print("exp_05 E3 -- RELEASE_GROUP paired A/B, arms interleaved in one "
          "process")
    print(f"  arms      : {', '.join(f'{t}(N={g})' for t, g in ARMS)}")
    print(f"  protocol  : {rounds} rounds x {iters} pipelined iters per arm, "
          f"rotating arm order, {warmup_ms} ms shared warmup per shape")
    print(f"  null arm  : {NULL_PAIR[0]} vs {NULL_PAIR[1]} -- same constant, "
          f"two modules; their delta is this instrument's noise floor")
    print("=" * 96)

    results = {}
    for shape in SCORED:
        results[f"{shape[0]}x{shape[1]}x{shape[2]}"] = {
            "shape": shape[:4],
            "arms": measure_shape(shape, rounds, iters, warmup_ms),
        }

    keys = list(results)
    base = NULL_PAIR[0]
    print("\n" + "=" * 96)
    print("PER-SHAPE SUMMARY (us; best and median of the interleaved rounds)")
    print("=" * 96)
    header = f"{'#':>2}{'m':>6}{'n':>6}{'k':>7}"
    for tag, _ in ARMS:
        header += f"{tag + ' best':>12}{tag + ' med':>11}"
    print(header)
    print("-" * len(header))
    for i, key in enumerate(keys):
        row = results[key]
        line = (f"{i+1:>2}{row['shape'][0]:>6}{row['shape'][1]:>6}"
                f"{row['shape'][2]:>7}")
        for tag, _ in ARMS:
            line += (f"{row['arms'][tag]['best']:>12.2f}"
                     f"{row['arms'][tag]['median']:>11.2f}")
        print(line)
    print("-" * len(header))

    print("\nGEOMETRIC MEANS (the competition's ranking statistic)")
    summary = {}
    for tag, group in ARMS:
        gm_best = geomean([results[k]["arms"][tag]["best"] for k in keys])
        gm_med = geomean([results[k]["arms"][tag]["median"] for k in keys])
        summary[tag] = {"release_group": group, "geomean_best": gm_best,
                        "geomean_median": gm_med}
    for tag, _ in ARMS:
        s = summary[tag]
        print(f"  {tag}: geomean best {s['geomean_best']:8.2f} us "
              f"(x{s['geomean_best'] / summary[base]['geomean_best']:.4f})   "
              f"geomean median {s['geomean_median']:8.2f} us "
              f"(x{s['geomean_median'] / summary[base]['geomean_median']:.4f})")

    # The null arm, per shape. Read this BEFORE any of the numbers above: an arm
    # delta smaller than |rg1 - rg1b| on the same shape is not a measurement of
    # release granularity, it is a measurement of this script.
    a, b = NULL_PAIR
    print(f"\nNULL ARM -- {a} vs {b} (identical binaries). Any real arm delta "
          f"must exceed this.")
    null_worst = 0.0
    for i, key in enumerate(keys):
        arms = results[key]["arms"]
        d_best = arms[b]["best"] / arms[a]["best"] - 1.0
        d_med = arms[b]["median"] / arms[a]["median"] - 1.0
        null_worst = max(null_worst, abs(d_best), abs(d_med))
        print(f"  {i+1}: {key:<22} best {d_best*100:+6.2f}%   "
              f"median {d_med*100:+6.2f}%")
    print(f"  worst null-arm separation: {null_worst*100:.2f}%")

    # A per-shape N is a different mechanism and would need its own ladder, but
    # the paired data prices it here: take each shape's own winner, and only
    # where its margin over rg1 clears the null arm on that shape.
    hybrid_best, hybrid_pick = [], []
    for key in keys:
        arms = results[key]["arms"]
        floor = abs(arms[b]["best"] / arms[a]["best"] - 1.0)
        real = [t for t, _ in ARMS if t != b
                and arms[t]["best"] / arms[a]["best"] - 1.0 < -floor]
        pick = min(real, key=lambda t: arms[t]["best"]) if real else a
        hybrid_best.append(arms[pick]["best"])
        hybrid_pick.append(f"{key.split('x')[0]}:{pick}")
    print(f"\n  per-shape winner where the margin clears that shape's null arm "
          f"(a bound, not a shipped config):\n    {geomean(hybrid_best):8.2f} us"
          f"  [{' '.join(hybrid_pick)}]")

    # Tagged, because run_ab.sh runs this several times with different arm
    # orders and an untagged name silently kept only the last one.
    tag = os.environ.get("AB_TAG", "run")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", f"ab_release_group_{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"rounds": rounds, "iters": iters, "arms": dict(ARMS),
                   "per_shape": results, "geomean": summary}, handle, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
