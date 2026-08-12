"""exp_26: per-shape release group, paired inside one process.

Descended from experiments/exp_05_release_granularity/ab_release_group.py, which
is the only instrument on this node that can resolve a release-granularity arm:
the gate ladder measures one arm per run, the run-to-run spread on an IDENTICAL
binary is larger than the effect on most shapes, and the harness bias is
per-allocation and partly allocation-ORDER. Here every arm is a separately
compiled module, all of them are open at once, and their timed blocks are
INTERLEAVED in a rotating order inside one round, so adjacent samples share the
clock state, the allocation state and the input sequence.

Arms:
  ps0   the shipped incumbent: rgroup = (tiles_per_cta >= 4) ? 4 : 1
  ps1   the candidate:         rgroup = max(1, min(4, tiles_per_cta))
  rg2c  RELEASE_GROUP=2 under the incumbent rule. On 8192x4096x14336 this
        computes rgroup = 2 by a different expression than ps1 does, so
        agreement between them there is behavioural evidence that ps1's rgroup
        really became 2. On 8192x8192x29568 it groups at 2 where ps0/ps1 group
        at 4, which prices the group size on the one shape that has a choice.
  ps0b  the NULL ARM: a second build of ps0. Nothing an instruction can see
        separates it from ps0, so |ps0b - ps0| is this instrument's own floor on
        that shape and no other arm may claim a delta inside it.

Reports best and median, never the mean: means are unusable on this node.

  python3 ab_pershape.py [rounds] [iters] [warmup_ms] [reverse]
"""

import json
import math
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import harness_lib as H                                   # noqa: E402
from harness_lib import rt, WORLD, GemmRS                 # noqa: E402

ARMS = os.environ.get("AB_ARMS", "ps0,ps1,ps2,rg2c,ps0b").split(",")
BASE = "ps0"
NULL_ARM = "ps0b"
CANDIDATE = os.environ.get("AB_CANDIDATE", "ps2")

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

TIGHT = 2e-3


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def per_cta(plan):
    tiles = int(plan["gemm_tiles"])
    producers = int(plan["num_gemm_ctas"])
    return tiles, producers, -(-tiles // producers)


def expected_rgroup(arm, ppc):
    """What the source computes for this arm at this tiles-per-CTA.

    Printed next to the numbers so a reader can see, without leaving the log,
    which arms are supposed to be the same instruction stream on this shape --
    those are the rows whose delta prices the instrument rather than the change.
    """
    if arm in ("ps0", "ps0b"):
        return 4 if ppc >= 4 else 1
    if arm == "ps1":
        return max(1, min(4, ppc))
    if arm == "ps2":
        return 4 if ppc >= 4 else (2 if ppc >= 2 else 1)
    if arm == "rg2c":
        return 2 if ppc >= 2 else 1
    return None


def measure_shape(shape, rounds, iters, warmup_ms, arms_order):
    m, n, k, bias, seed = shape
    arms = {}
    try:
        for tag in arms_order:
            h = GemmRS(m, n, k, bias, module_name=f"gemm_rs_mi300x_{tag}")
            h.set_inputs(seed)
            arms[tag] = h
        plan = arms[arms_order[0]].plan
        tiles, producers, ppc = per_cta(plan)
        print(f"\n=== m={m} n={n} k={k} bias={int(bias)} "
              f"(row {int(plan['config_row'])}) ===")
        print(f"    {tiles} tiles over {producers} producer CTAs "
              f"= up to {ppc} tiles per CTA")
        print("    expected rgroup: " + "  ".join(
            f"{t}={expected_rgroup(t, ppc)}" for t in ARMS)
            + ("   <-- built-in control: every arm collapses to rgroup 1"
               if ppc == 1 else ""))

        # Correctness and cross-arm bit-exactness before any timing. A release
        # granularity change must not move a single bit of arithmetic; this is a
        # stronger statement than 2e-3 and it is free here, because every arm is
        # driven over the same inputs in the same process.
        ref = None
        for tag in arms_order:
            h = arms[tag]
            h.launch()
            checks = h.verify(rtol=TIGHT, atol=TIGHT)
            assert all(c["allclose"] for c in checks), f"{tag} failed {TIGHT}"
            assert not h.error_report(), f"{tag} raised {h.error_report()}"
            if ref is None:
                ref = [h.out[r].clone() for r in range(WORLD)]
            else:
                same = all(torch.equal(h.out[r], ref[r]) for r in range(WORLD))
                assert same, f"{tag} is NOT bit-identical to {arms_order[0]}"
        print(f"    all {len(arms_order)} arms correct at {TIGHT} and "
              f"bit-identical to each other on all 8 ranks")

        # Shared duration-based warmup: idle sclk is ~120-132 MHz against ~1900
        # under load, and a fixed-iteration warmup once produced a 60% wrong
        # number on this node.
        started = time.perf_counter()
        while (time.perf_counter() - started) * 1e3 < warmup_ms:
            for tag in arms_order:
                arms[tag].launch(sync=False)
            for tag in arms_order:
                arms[tag].sync()

        # Arm order WITHIN a round: shuffled, not rotated.
        #
        # The rotation this instrument inherited from exp_05 advances every arm
        # by one position per round, which balances absolute position but pins
        # the RELATIVE spacing of any two arms: with `order = (r + i) % n`, arm
        # j always runs exactly (j - k) mod n blocks after arm k, in every round
        # of every pass. Any effect that depends on what ran just before a block
        # therefore does not average out between a given pair -- it is a
        # constant offset added to that pair's paired difference. The null arm
        # measured it: ps0b, which differs from ps0 in nothing an instruction
        # can see, came out -2.29% against ps0 on 8192x8192x29568 and -0.77% on
        # 4096x4096x4096, in 57 and 58 of 64 paired rounds. A permutation drawn
        # per round makes the spacing between any two arms vary, so that offset
        # averages out instead of accumulating.
        #
        # Seeded from the shape so the sequence is reproducible and independent
        # of arms_order -- the forward and reversed passes must not receive
        # correlated permutations, or the reversal stops being an independent
        # draw.
        rng = random.Random(0xC0FFEE ^ (m * 1315423911) ^ (n << 7) ^ k)
        samples = {tag: [] for tag in arms_order}
        for r in range(rounds):
            shuffled = list(arms_order)
            rng.shuffle(shuffled)
            for tag in shuffled:
                wall, _ = arms[tag]._timed_block(iters)
                samples[tag].append(wall)

        out = {}
        for tag in arms_order:
            values = samples[tag]
            out[tag] = {
                "expected_rgroup": expected_rgroup(tag, ppc),
                "best": min(values),
                "median": statistics.median(values),
                "worst": max(values),
                "spread_pct": (max(values) - min(values)) / min(values) * 100.0,
                "samples": values,
            }
        for tag in arms_order:
            row = out[tag]
            print(f"    {tag:<5} best {row['best']:9.2f}  "
                  f"median {row['median']:9.2f}  "
                  f"spread {row['spread_pct']:5.2f}%  "
                  f"best x{row['best'] / out[BASE]['best']:.4f}  "
                  f"median x{row['median'] / out[BASE]['median']:.4f}")
        for tag in arms_order:
            assert not arms[tag].error_report(), f"{tag} raised late"
        out["_geometry"] = {"tiles": tiles, "producers": producers,
                            "tiles_per_cta": ppc,
                            "config_row": int(plan["config_row"])}
        return out
    finally:
        for h in arms.values():
            h.close()


def main():
    rt.enable_peer_access(WORLD)
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    warmup_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    arms_order = list(ARMS)
    reverse = len(sys.argv) > 4 and sys.argv[4] not in ("0", "", "no")
    if reverse:
        # Reversing the arm list reverses ALLOCATION order too, which is the
        # other half of the null arm's job: the harness bias on this node is
        # per-allocation and partly allocation-order, so a delta that survives
        # both directions is not a positional artefact.
        arms_order = list(reversed(arms_order))

    print("=" * 100)
    print("exp_26 -- per-shape release group, arms interleaved in one process")
    print(f"  arms      : {', '.join(arms_order)}"
          f"{'   (REVERSED allocation order)' if reverse else ''}")
    print(f"  protocol  : {rounds} rounds x {iters} pipelined iters per arm, "
          f"arm order SHUFFLED per round, {warmup_ms} ms warmup per shape")
    print(f"  null arm  : {BASE} vs {NULL_ARM} -- same rule, two modules; their "
          f"delta is this instrument's floor")
    print("=" * 100)

    results = {}
    for shape in SCORED:
        results[f"{shape[0]}x{shape[1]}x{shape[2]}"] = {
            "shape": shape[:4],
            "arms": measure_shape(shape, rounds, iters, warmup_ms, arms_order),
        }

    keys = list(results)
    print("\n" + "=" * 100)
    print("PER-SHAPE SUMMARY (us; best and median of the interleaved rounds)")
    print("=" * 100)
    header = f"{'#':>2}{'m':>6}{'n':>6}{'k':>7}{'t/CTA':>6}"
    for tag in ARMS:
        header += f"{tag + ' best':>12}{tag + ' med':>11}"
    print(header)
    print("-" * len(header))
    for i, key in enumerate(keys):
        row = results[key]
        line = (f"{i+1:>2}{row['shape'][0]:>6}{row['shape'][1]:>6}"
                f"{row['shape'][2]:>7}"
                f"{row['arms']['_geometry']['tiles_per_cta']:>6}")
        for tag in ARMS:
            line += (f"{row['arms'][tag]['best']:>12.2f}"
                     f"{row['arms'][tag]['median']:>11.2f}")
        print(line)
    print("-" * len(header))

    print("\nGEOMETRIC MEANS (the competition's ranking statistic)")
    summary = {}
    for tag in ARMS:
        gm_best = geomean([results[k]["arms"][tag]["best"] for k in keys])
        gm_med = geomean([results[k]["arms"][tag]["median"] for k in keys])
        summary[tag] = {"geomean_best": gm_best, "geomean_median": gm_med}
    for tag in ARMS:
        s = summary[tag]
        print(f"  {tag:<5} geomean best {s['geomean_best']:8.2f} us "
              f"(x{s['geomean_best'] / summary[BASE]['geomean_best']:.4f})   "
              f"geomean median {s['geomean_median']:8.2f} us "
              f"(x{s['geomean_median'] / summary[BASE]['geomean_median']:.4f})")

    # The null arm, per shape. Read this BEFORE any of the numbers above.
    print(f"\nNULL ARM -- {BASE} vs {NULL_ARM} (identical rule, two builds). "
          f"Any real arm delta must exceed this.")
    null_worst = 0.0
    for i, key in enumerate(keys):
        arms = results[key]["arms"]
        d_best = arms[NULL_ARM]["best"] / arms[BASE]["best"] - 1.0
        d_med = arms[NULL_ARM]["median"] / arms[BASE]["median"] - 1.0
        null_worst = max(null_worst, abs(d_best), abs(d_med))
        print(f"  {i+1}: {key:<22} best {d_best*100:+6.2f}%   "
              f"median {d_med*100:+6.2f}%")
    print(f"  worst null-arm separation: {null_worst*100:.2f}%")

    # The candidate against the incumbent, priced on each shape's own floor.
    print(f"\nCANDIDATE {CANDIDATE} vs {BASE}, against that shape's null arm")
    verdicts = {}
    for i, key in enumerate(keys):
        arms = results[key]["arms"]
        floor = abs(arms[NULL_ARM]["best"] / arms[BASE]["best"] - 1.0)
        d_best = arms[CANDIDATE]["best"] / arms[BASE]["best"] - 1.0
        d_med = arms[CANDIDATE]["median"] / arms[BASE]["median"] - 1.0
        # Disjoint ranges are the standard this session requires: a delta whose
        # sample ranges overlap is not separated, whatever the point estimates.
        lo_c, hi_c = (min(arms[CANDIDATE]["samples"]),
                      max(arms[CANDIDATE]["samples"]))
        lo_b, hi_b = min(arms[BASE]["samples"]), max(arms[BASE]["samples"])
        disjoint = hi_c < lo_b or hi_b < lo_c
        same_stream = (arms[CANDIDATE]["expected_rgroup"]
                       == arms[BASE]["expected_rgroup"])
        verdict = ("control (same rgroup)" if same_stream else
                   ("WIN" if d_best < -floor and disjoint else
                    ("LOSS" if d_best > floor and disjoint else "inside floor")))
        verdicts[key] = {"delta_best_pct": d_best * 100,
                         "delta_median_pct": d_med * 100,
                         "floor_pct": floor * 100, "disjoint": disjoint,
                         "same_stream": same_stream, "verdict": verdict}
        print(f"  {i+1}: {key:<22} best {d_best*100:+6.2f}%  "
              f"median {d_med*100:+6.2f}%  floor {floor*100:5.2f}%  "
              f"ranges {'disjoint' if disjoint else 'overlap  '}  {verdict}")

    tag = os.environ.get("AB_TAG", "run")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", f"ab_pershape_{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"rounds": rounds, "iters": iters, "arms": arms_order,
                   "reversed": reverse, "per_shape": results,
                   "geomean": summary, "candidate_vs_base": verdicts},
                  handle, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
