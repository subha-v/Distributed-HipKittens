#!/usr/bin/env python3
"""exp_22 pre-flight: is the node still in the state we think it is?

exp_22 is the first campaign to run after the exp_26 M9 run took a
VM_L2_PROTECTION_FAULT and left a stale KFD entry holding ~1.25 GB per GPU with
zero CU occupancy.  "Expected timing effect: nil" is a hypothesis, and the
cheapest way to test it is to re-measure a value we already know.

WHICH value, and against WHICH statistic, is the whole difficulty.  The first
version of this gate compared M7's numbers to the best-of-arm vector
`62.38 / 64.52 / 83.75 / 198.71 / 613.70 / 1616.63` and failed shape 5 at
+5.3%.  That is precisely the trap LESSONS.md:189 records:

    the recorded per-shape denominator vector is BEST-of-arm, while gate M7
    prints MEANS ... exp_05's own table records shape 5 as 613.70 / 645.93 =
    best / median.  Comparing an M7 mean geomean to it manufactured a phantom
    "+7.3% regression on shape 5" for a row whose configuration and device code
    had not changed -- shape 5 measured 644.71-669.39 across seven independent
    allocations, i.e. exactly its recorded median.

A best over three rotations is also a weaker order statistic than a best pooled
over a multi-arm campaign, so every shape's best drifts up by a few percent for
purely combinatorial reasons.  So the gate compares like with like:

  PRIMARY   geomean of M7 means vs 207.18 us, the mean-geomean recorded when
            E4b landed the current best (LESSONS.md:598).  Same statistic, same
            protocol, same rotation count.
  SHAPE 5   mean vs the recorded MEDIAN 645.93 us, and membership of the
            documented same-configuration range 644.71-669.39 us.
  CONTEXT   bests against the best-of-arm vector, reported but NOT gated,
            because best-vs-best across sessions is documented-fragile for any
            shape with a fat lower tail, which shape 5 has.

  sanity_check.py m7_results.json [--json sanity.json]
"""

import argparse
import json
import math
import sys

SHAPES = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
          "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]
# Best-of-arm, us (LESSONS.md:600). Context only -- see the docstring.
REFERENCE_BEST = [62.38, 64.52, 83.75, 198.71, 613.70, 1616.63]
# Gated, like-for-like statistics.
REFERENCE_MEAN_GEOMEAN = 207.18          # LESSONS.md:598, M7 means at the E4b landing
S5_MEDIAN = 645.93                       # exp_05 table, shape 5 best/median = 613.70/645.93
S5_SAME_CONFIG_RANGE = (644.71, 669.39)  # seven independent allocations, LESSONS.md:194
GATE_INDEX = 4


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json", default="sanity.json")
    ap.add_argument("--geomean-tolerance", type=float, default=0.03)
    ap.add_argument("--shape5-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.results) as handle:
        doc = json.load(handle)
    samples = doc["samples_us"]
    means = doc["means_us"]

    rows = []
    header = (f"{'#':>2} {'shape':<18}{'mean us':>10}{'best us':>10}"
              f"{'ref best':>10}{'d(best)':>10}{'correct':>9}")
    print(header)
    print("-" * len(header))
    for index in range(len(SHAPES)):
        values = samples[str(index)]
        best, mean = min(values), means[index]
        delta = (best - REFERENCE_BEST[index]) / REFERENCE_BEST[index]
        detail = doc.get("detail", {}).get(str(index), {})
        correct = detail.get("correct_tight", detail.get("correct"))
        rows.append({
            "shape": SHAPES[index], "mean_us": mean, "best_us": best,
            "rotation_samples_us": values,
            "reference_best_us": REFERENCE_BEST[index],
            "rel_delta_best_context_only": delta, "correct_tight": correct,
        })
        print(f"{index+1:>2} {SHAPES[index]:<18}{mean:>10.2f}{best:>10.2f}"
              f"{REFERENCE_BEST[index]:>10.2f}{delta:>+9.2%}{str(correct):>9}")
    print("  (the best column is CONTEXT: a best over 3 rotations is a weaker "
          "order statistic\n   than a best pooled over a multi-arm campaign, "
          "so it drifts up for free.)")

    gm = geomean(means)
    gm_delta = (gm - REFERENCE_MEAN_GEOMEAN) / REFERENCE_MEAN_GEOMEAN
    gm_ok = abs(gm_delta) <= args.geomean_tolerance

    s5_mean = means[GATE_INDEX]
    s5_delta = (s5_mean - S5_MEDIAN) / S5_MEDIAN
    s5_ok = abs(s5_delta) <= args.shape5_tolerance
    s5_in_range = S5_SAME_CONFIG_RANGE[0] <= s5_mean <= S5_SAME_CONFIG_RANGE[1]

    all_correct = bool(doc.get("all_correct"))

    print()
    print(f"PRIMARY  geomean of means {gm:8.2f} us vs recorded "
          f"{REFERENCE_MEAN_GEOMEAN:.2f} us : {gm_delta:+.2%}  "
          f"{'ok' if gm_ok else 'FAIL'}")
    print(f"SHAPE 5  mean            {s5_mean:8.2f} us vs recorded median "
          f"{S5_MEDIAN:.2f} us : {s5_delta:+.2%}  {'ok' if s5_ok else 'FAIL'}")
    print(f"SHAPE 5  inside the documented same-config range "
          f"{S5_SAME_CONFIG_RANGE[0]}-{S5_SAME_CONFIG_RANGE[1]} us : "
          f"{'yes' if s5_in_range else 'NO'}")
    print(f"CORRECT  all six shapes at 2e-3 : {all_correct}")

    verdict = "PASS" if (gm_ok and s5_ok and all_correct) else "FAIL"
    print(f"NODE SANITY: {verdict}")
    if verdict == "FAIL":
        print("  STOP. The node is not in the state the ratchet was measured "
              "in. This matters more than the figure -- report before "
              "collecting any timeline data.")

    with open(args.json, "w") as handle:
        json.dump({
            "schema": "exp22.sanity.v2",
            "context": "first campaign after the exp_26 VM_L2_PROTECTION_FAULT "
                       "left a stale KFD entry holding ~1.25 GB/GPU with zero "
                       "CU occupancy",
            "gate": {
                "primary": "geomean of M7 means vs 207.18 us (LESSONS.md:598)",
                "shape5": "M7 mean vs recorded median 645.93 us, and "
                          "membership of 644.71-669.39 us (LESSONS.md:194)",
                "not_gated": "best-of-arm vector; best-vs-best across sessions "
                             "is documented-fragile (LESSONS.md:189)",
            },
            "verdict": verdict,
            "geomean_of_means_us": gm,
            "geomean_reference_us": REFERENCE_MEAN_GEOMEAN,
            "geomean_rel_delta": gm_delta,
            "shape5_mean_us": s5_mean,
            "shape5_reference_median_us": S5_MEDIAN,
            "shape5_rel_delta": s5_delta,
            "shape5_in_same_config_range": s5_in_range,
            "all_correct_tight": all_correct,
            "rows": rows,
        }, handle, indent=2)
    print(f"wrote {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
