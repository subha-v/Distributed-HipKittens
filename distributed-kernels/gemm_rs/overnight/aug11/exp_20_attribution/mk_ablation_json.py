"""Turn a reattribute log into aug11/exp_20_attribution/ablation.json.

The driver (harness/exp_ablation.py) prints a fixed-width table and nothing
machine-readable, so this parses that table. Three deliberate choices:

  * Columns are read by FIXED-WIDTH SLICE, not by whitespace split. The driver's
    delta columns are 8 characters wide and a large negative delta ("-1234.5"
    plus sign) can touch its neighbour, which silently fuses two fields under a
    whitespace split. The six ARM columns are 9 wide and are the only fields
    read here.
  * The five stage figures are RECOMPUTED from the arm timings rather than
    parsed, so the JSON cannot disagree with its own arms.
  * Every stage delta carries the shape's allocation-noise floor in absolute us
    and a boolean saying whether the delta clears it. Ablation cells are
    separate processes with separate allocations and the harness bias on this
    node is per-allocation, so a delta under its floor is not a measurement.

Usage: mk_ablation_json.py <reattribute log> <out.json> [key=value ...]
"""

import json
import re
import sys
import time

# (index, m, n, k, has_bias, bm, bn, bk, nr) in the driver's row order.
# Tile/reducer columns are the live table at gemm_rs_mi300x_host_abi.hpp:109-116.
SHAPES = [
    (1, 64, 7168, 18432, False, 32, 64, 128, 56),
    (2, 512, 4096, 12288, True, 64, 128, 64, 32),
    (3, 2048, 2880, 2880, True, 128, 192, 32, 32),
    (4, 4096, 4096, 4096, False, 256, 256, 32, 32),
    (5, 8192, 4096, 14336, True, 256, 256, 32, 32),
    (6, 8192, 8192, 29568, False, 256, 256, 32, 48),
]

# Per-shape null-arm floor, percent of `full`. HANDOFF.md:102-105 publishes
# 1.34/0.56/0.61/2.41/2.17/4.28 and exp_14 later measured 3.48/0.93/1.61 on
# shapes 1-3 over 15 draws, so the larger of the two is used and the whole set
# is treated as a LOWER bound.
FLOOR_PCT = {1: 3.48, 2: 0.93, 3: 1.61, 4: 2.41, 5: 2.17, 6: 4.28}

# Arm column order as printed, and the stage each single cut prices.
ARMS = ["full", "nomain", "emitlocal", "nored", "noproto", "norelease"]
STAGES = [("gemm", "nomain"), ("egress", "emitlocal"), ("reduce", "nored"),
          ("sync", "noproto"), ("release", "norelease")]

SHAPE_W = 20
ARM_W = 9

SCHEMA = {
    "unit": "microseconds, pipelined wall time per world-8 operation, "
            "40 iterations per cell",
    "shapes[].full_us": "unablated scratch build of the current kernel; the "
                        "denominator for every share",
    "shapes[].arms_us": "raw pipelined wall time of each single-cut arm; every "
                        "stage below is full_us minus one of these",
    "shapes[].stages_us.gemm": "full - nomain: the k-loop's MFMA math plus its "
                               "A/B global->LDS traffic",
    "shapes[].stages_us.egress": "full - emitlocal: the peer/xGMI cost of "
                                 "egress with store volume and instruction "
                                 "count held fixed (same bytes to the local "
                                 "rank's own slot)",
    "shapes[].stages_us.reduce": "full - nored: the reducer's 8-source pull, "
                                 "fp32 sum and bf16 store",
    "shapes[].stages_us.sync": "full - noproto: all cross-rank synchronization "
                               "(credit waits, ready publishes, reducer "
                               "participation)",
    "shapes[].stages_us.release": "full - norelease: the per-release-group "
                                  "buffer_wbl2 sc0 sc1 L2 writeback plus its "
                                  "vmcnt drain",
    "shapes[].stage_share": "stages_us / full_us",
    "shapes[].floor_pct / floor_us": "the shape's allocation-noise floor. A "
                                     "stage whose |delta| is under floor_us is "
                                     "NOT measured; see stage_clears_floor.",
    "shapes[].stage_clears_floor": "per stage, |delta| >= floor_us",
    "shapes[].ranked_stages": "stages by absolute us, largest first; entries "
                              "that do not clear the floor are still listed "
                              "but must not be ranked against each other",
    "shapes[].stage_sum_over_full": "sum of the five stages / full. Single-cut "
                                    "deltas OVERLAP wherever the phases "
                                    "overlap, so this is NOT expected to be "
                                    "1.0 and the stages are NOT a budget: >1 "
                                    "means the phases genuinely overlap in the "
                                    "pipeline, <1 means part of full is priced "
                                    "by no cut.",
    "ranking": "stages ordered by absolute us on shape 6 (the largest graded "
               "shape); also carries each stage's summed us over all six "
               "shapes and its mean share.",
}

CAVEATS = [
    "Only the `full` arm is numerically correct; the other five exist solely "
    "to price a phase and are expected to fail verification.",
    "Deltas are single-cut and therefore NOT additive. Rank stages by size; "
    "do not treat the sum as a budget.",
    "`norelease` removes publication ordering and is not a legal kernel.",
    "One process drives all 8 devices here, which is not the evaluator's "
    "topology.",
    "Shape 1 is bound=HOST at ~10x SOL: a ~63 us call whose stage deltas are a "
    "few us and are inside its floor. No mainloop conclusion may be drawn "
    "from it.",
]


def parse_row(line):
    """Read the six arm timings out of one fixed-width table row."""
    body = line[SHAPE_W:]
    values = []
    for i in range(len(ARMS)):
        field = body[i * ARM_W:(i + 1) * ARM_W].strip()
        values.append(float(field) if field else float("nan"))
    return values


def main():
    log_path = sys.argv[1]
    out_path = sys.argv[2]
    extra = dict(kv.split("=", 1) for kv in sys.argv[3:] if "=" in kv)

    text = open(log_path, errors="replace").read()

    rows, notes = {}, {}
    current = None
    for line in text.splitlines():
        match = re.match(r"^(\d+)x(\d+)x(\d+)\s", line)
        if match:
            key = match.group(0).strip()
            rows[key] = parse_row(line)
            current = key
            continue
        if current and line.startswith("    ") and line.strip():
            if "=" not in line or "FAULT" in line or "FAIL" in line:
                notes.setdefault(current, []).append(line.strip())

    shapes_out, missing = [], []
    for index, m, n, k, bias, bm, bn, bk, nr in SHAPES:
        key = f"{m}x{n}x{k}"
        if key not in rows:
            missing.append(key)
            continue
        arms = dict(zip(ARMS, rows[key]))
        full = arms["full"]
        floor_pct = FLOOR_PCT[index]
        floor_us = full * floor_pct / 100.0
        stages = {stage: round(full - arms[arm], 2) for stage, arm in STAGES}
        record = {
            "shape_index": index,
            "shape": key,
            "m": m, "n": n, "k": k, "has_bias": bias,
            "bm": bm, "bn": bn, "bk": bk, "nr": nr,
            "full_us": round(full, 2),
            "arms_us": {a: round(v, 2) for a, v in arms.items() if a != "full"},
            "stages_us": stages,
            "stage_share": {s: round(v / full, 4) for s, v in stages.items()},
            "floor_pct": floor_pct,
            "floor_us": round(floor_us, 2),
            "stage_clears_floor": {s: bool(abs(v) >= floor_us)
                                   for s, v in stages.items()},
            "ranked_stages": [s for s, _ in sorted(stages.items(),
                                                   key=lambda kv: -kv[1])],
            "stage_sum_us": round(sum(stages.values()), 2),
            "stage_sum_over_full": round(sum(stages.values()) / full, 4),
        }
        if key in notes:
            record["notes"] = notes[key]
        shapes_out.append(record)

    by_index = {r["shape_index"]: r for r in shapes_out}
    ranking = []
    for stage, _ in STAGES:
        present = [r for r in shapes_out]
        entry = {
            "stage": stage,
            "shape6_us": by_index[6]["stages_us"][stage] if 6 in by_index
            else None,
            "shape6_share": by_index[6]["stage_share"][stage] if 6 in by_index
            else None,
            "sum_us_all_shapes": round(
                sum(r["stages_us"][stage] for r in present), 2),
            "mean_share": round(
                sum(r["stage_share"][stage] for r in present) / len(present), 4),
            "clears_floor_on_shapes": [r["shape_index"] for r in present
                                       if r["stage_clears_floor"][stage]],
        }
        ranking.append(entry)
    ranking.sort(key=lambda e: -(e["shape6_us"] or 0))

    doc = {
        "experiment": "aug11/exp_20_attribution",
        "what": "macro-gated stage attribution of the fused GEMM->ReduceScatter "
                "megakernel at the current best config",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_log": log_path,
        "iters": int(extra.pop("iters", 40)),
        "config": {
            "wgm4": 0,
            "release_group": 4,
            "release_group_full_only": 1,
            "grid_ctas": 304,
            "world": 8,
            "arch": "gfx942 (MI300X, SPX, 304 CU)",
            "note": "per-shape bm/bn/bk/nr are on each shape record; "
                    "host_abi.hpp:109-116",
            **{k: v for k, v in extra.items()
               if not k.startswith("clocks_")},
        },
        "clocks": {
            "before": extra.get("clocks_before", ""),
            "after": extra.get("clocks_after", ""),
        },
        "freshness_gate": {
            "expected_shape6_full_us": 1617,
            "tolerance_pct": 12,
            "measured_shape6_full_us": by_index[6]["full_us"]
            if 6 in by_index else None,
            "passed": (abs(by_index[6]["full_us"] - 1617) / 1617 <= 0.12)
            if 6 in by_index else False,
        },
        "noise_floor_pct": FLOOR_PCT,
        "schema": SCHEMA,
        "caveats": CAVEATS,
        "shapes": shapes_out,
        "ranking": ranking,
    }
    if missing:
        doc["missing_shapes"] = missing

    with open(out_path, "w") as handle:
        json.dump(doc, handle, indent=2)

    print(f"wrote {out_path} with {len(shapes_out)} shapes"
          + (f"; MISSING {missing}" if missing else ""))
    print(f"{'shape':<18}{'full':>9}" + "".join(f"{s:>9}" for s, _ in STAGES)
          + f"{'floor':>8}{'sum/full':>10}  ranking (* = under floor)")
    for record in shapes_out:
        marks = "".join(
            f"{record['stages_us'][s]:>8.1f}"
            + ("*" if not record["stage_clears_floor"][s] else " ")
            for s, _ in STAGES)
        print(f"{record['shape']:<18}{record['full_us']:>9.1f}{marks}"
              f"{record['floor_us']:>8.1f}{record['stage_sum_over_full']:>10.2f}"
              "  " + " > ".join(record["ranked_stages"]))
    print("\nranking by absolute us on shape 6:")
    for entry in ranking:
        print(f"  {entry['stage']:<9}{entry['shape6_us']:>9.1f} us "
              f"({entry['shape6_share'] * 100:5.1f}% of full)  "
              f"clears floor on shapes {entry['clears_floor_on_shapes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
