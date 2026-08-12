#!/usr/bin/env python3
"""exp_22 pre-flight: is the node still in the state we think it is?

exp_22 is the first campaign to run after the exp_26 M9 run took a
VM_L2_PROTECTION_FAULT and left a stale KFD entry holding ~1.25 GB per GPU.
That entry has zero CU occupancy so the expected timing effect is nil -- but
"expected nil" is a hypothesis, and the cheapest way to test it is to
re-measure a value we already know.

Reads harness/m7_bench.py's m7_results.json and compares the six graded shapes
against the current best-of-arm vector.  Shape 5 is the gate, because it is the
shape this figure is about; the other five are reported for context.  A failure
here matters far more than the figure: it means the node is not what we think,
and no number taken afterwards is trustworthy.

  sanity_check.py m7_results.json [--json sanity.json] [--tolerance 0.05]
"""

import argparse
import json
import statistics
import sys

# Current best-of-arm, us, per aug11 STATUS/LESSONS.
REFERENCE = [62.38, 64.52, 83.75, 198.71, 613.70, 1616.63]
SHAPES = ["64x7168x18432", "512x4096x12288", "2048x2880x2880",
          "4096x4096x4096", "8192x4096x14336", "8192x8192x29568"]
GATE_INDEX = 4          # shape 5, the figure's shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json", default="sanity.json")
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.results) as handle:
        doc = json.load(handle)
    samples = doc["samples_us"]
    means = doc["means_us"]

    rows, gate_ok = [], True
    header = (f"{'#':>2} {'shape':<18}{'best us':>10}{'mean us':>10}"
              f"{'ref best':>10}{'d(best)':>10}{'correct':>9}")
    print(header)
    print("-" * len(header))
    for index in range(len(REFERENCE)):
        values = samples[str(index)]
        best = min(values)
        mean = means[index]
        delta = (best - REFERENCE[index]) / REFERENCE[index]
        detail = doc.get("detail", {}).get(str(index), {})
        correct = detail.get("correct_tight", detail.get("correct"))
        rows.append({
            "shape": SHAPES[index], "best_us": best, "mean_us": mean,
            "rotation_samples_us": values,
            "reference_best_us": REFERENCE[index], "rel_delta": delta,
            "correct_tight": correct,
        })
        flag = ""
        if index == GATE_INDEX:
            gate_ok = abs(delta) <= args.tolerance
            flag = "  <== GATE" + ("" if gate_ok else "  FAIL")
        print(f"{index+1:>2} {SHAPES[index]:<18}{best:>10.2f}{mean:>10.2f}"
              f"{REFERENCE[index]:>10.2f}{delta:>+9.2%}{str(correct):>9}{flag}")

    all_correct = doc.get("all_correct")
    geomean = doc.get("geomean_us")
    print()
    print(f"geomean (means, pipelined wall): {geomean:.2f} us")
    print(f"all shapes correct at 2e-3    : {all_correct}")
    verdict = "PASS" if (gate_ok and all_correct) else "FAIL"
    print(f"NODE SANITY (shape 5 within {args.tolerance:.0%} of "
          f"{REFERENCE[GATE_INDEX]} us AND all shapes correct): {verdict}")
    if verdict == "FAIL":
        print("  STOP. The node is not in the state the ratchet was measured "
              "in. This matters more than the figure -- report before "
              "collecting any timeline data.")

    with open(args.json, "w") as handle:
        json.dump({
            "schema": "exp22.sanity.v1",
            "context": "first campaign after the exp_26 VM_L2_PROTECTION_FAULT "
                       "left a stale KFD entry holding ~1.25 GB/GPU with zero "
                       "CU occupancy",
            "tolerance": args.tolerance,
            "gate_shape": SHAPES[GATE_INDEX],
            "verdict": verdict,
            "geomean_us": geomean,
            "all_correct_tight": all_correct,
            "rows": rows,
        }, handle, indent=2)
    print(f"wrote {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
