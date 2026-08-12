"""exp_24 aggregator: every arm's output -> ladders.json.

Two inputs, two shapes of work:

  * Instrument A (`raw/ladder/lad_s<i>.rank<r>.json`) is already JSON -- pool the
    eight ranks per arm per protocol and reduce.
  * Instrument B is the official evaluator, and all three of its drivers emit
    popcorn/stdout TEXT and no JSON at all, so that parse is real work. It is
    written to fail loudly: a run that emitted `benchmark-count: 6` and then hit
    its wall after three shapes (which is exactly what rank-1 did in exp_10) must
    raise, not silently produce three nulls and a geomean over half a ladder.

`--fixture` parses saved historical popcorn output and prints the table without
touching a GPU. That is the cheapest honest way to validate a text parser.

Usage:
  python3 ladders.py --root <exp dir> --out ladders.json
  python3 ladders.py --fixture                  # the three saved compbench runs
  python3 ladders.py --fixture <path> [...]     # specific files
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

ON = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight"

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]
SHAPE_LABELS = [f"{m}x{n}x{k}" for m, n, k, _, _ in SCORED]

RATIO_ARMS = ["ours", "reference", "rank1"]
NULL_ARMS = ["ours_null", "harness_floor"]
ALL_ARMS = RATIO_ARMS + NULL_ARMS
PROTOCOLS = ["graded", "pipelined"]

# Reproduced verbatim in result.md. Every one of these decides whether the
# numbers mean anything.
CAVEATS = [
    "TOPOLOGY MISMATCH. Our harness runs ONE process driving 8 devices with peer "
    "access; the evaluator's topology is ONE PROCESS PER RANK with "
    "torch.distributed. That is address-space equivalent for this kernel (the "
    "descriptor holds each rank's own base, and translate_peer only computes "
    "peer_base + (local_ptr - local_base)), but it is NOT the evaluator's "
    "topology. This applies to every comparison against rank-1's evaluator "
    "numbers.",

    "~92 us OF EVERY GRADED CALL IS HARNESS MACHINERY BOTH ARMS PAY. The "
    "evaluator's own trailing synchronize + barrier measures 57-79 us with an "
    "EMPTY timed region, plus 15-20 us of input clone; our own host path is "
    "~5-7 us against a 4.86 us floor. The graded ladder is reported BOTH raw and "
    "with that constant netted out, labelled. Netting it out makes our true "
    "kernel-to-kernel ratio WORSE than the headline, and it is why the small "
    "shapes are +/-10% noisy regardless of the kernel. This run measures the "
    "constant itself via the harness_floor arm instead of quoting it.",

    "rank-1 NEEDS AMDGCN_USE_BUFFER_OPS=0 (disclosed repair #6). Triton 3.6.0 "
    "lowers its peer stores to buffer_store_dwordx2, whose voffset is 32-bit, "
    "truncating element offsets of -6.6e8 ... -4.4e9. TRITON_CACHE_DIR must be "
    "cleared whenever that knob changes, because the cached hsaco has "
    "buffer_store baked in and leaving the cache makes the knob look "
    "ineffective. Argued behaviour-preserving in exp_10 result.md 2.6: the "
    "kernel cannot be correct at all without 64-bit peer addressing. Residual "
    "bias runs AGAINST rank-1 (its A/B operand loads lose buffer-op lowering "
    "too), so it cannot manufacture a win for us.",

    "rank-1's WARM PASS IS MANDATORY. The evaluator arm runs benchmark(warm) -> "
    "test -> benchmark(bench) in that order because eval.py's test mode "
    "hardcodes a 60 s per-rank timeout and a cold compile of rank-1's kernel "
    "exceeds it. The pass order is not simplified.",

    "THE REFERENCE ARM'S ABSOLUTE NUMBERS ARE JUNK; only same-instrument ratios "
    "are meaningful. Its own header records relative standard deviations of "
    "12-93%. Quote the ratio, never the absolute.",

    "rocm-smi --showpids PREFLIGHT RUNS FOR ALL THREE ARMS. Only "
    "run_reference_arm.sh does this itself; run_ours_evaluator.sh and the rank-1 "
    "driver do not, so the orchestrator adds it plus a KFD-fd drain wait.",

    "THE PUBLISHED rank-1 SCORE OF 413.139 us IS ON A DIFFERENT MACHINE UNDER A "
    "DIFFERENT PROTOCOL and is not comparable to anything measured here. The only "
    "valid denominator is rank-1 measured on this node.",

    "tools/run_rank1_bench3.sh CANNOT RUN rank-1 ON THIS NODE and was not used. "
    "It predates exp_10's repair #6: it omits AMDGCN_USE_BUFFER_OPS=0 (and builds "
    "its own ENVS string, so the knob cannot be injected from outside), points "
    "PYTHONPATH at a $ON/compat that does not exist so repair #3's sitecustomize "
    "is absent, and invokes $ON/patch_rank1.py which does not exist either (the "
    "tool is at $ON/tools/patch_rank1.py). The rank-1 evaluator arm is driven by "
    "experiments/exp_10_rank1/r1_eval.sh, which is the driver that produced "
    "exp_10 section 4, keeping bench3's warm -> test -> bench pass order. "
    "Nothing under tools/ was edited.",

    "MEANS ALONE ARE UNUSABLE ON THIS NODE even same-run interleaved: the bias is "
    "per-allocation and partly allocation-ORDER, not positional. exp_14 measured "
    "4.28% between IDENTICAL arms on shape 6 while positional residual stayed "
    "under +/-0.47%. Best and median are reported alongside every mean, and the "
    "ours_null arm measures that spread in this same run.",

    "THE RATIO'S OWN NOISE FLOOR IS +/-2%, measured in exp_14 on rows 4/5/6 whose "
    "configuration did not change between runs. A ratio change smaller than 2% is "
    "not reported as a change.",

    "BIAS IS FORCED ON FOR ALL SIX SHAPES, FOR ALL ARMS IDENTICALLY, because that "
    "is what the official evaluator actually does: its cases parser does "
    "int(val) and keeps the raw string on ValueError, so `has_bias: False` "
    "becomes the truthy STRING \"False\". rank-1's cached fast path also "
    "dereferences bias unconditionally and raises on its second call without "
    "one. A genuine has_bias=False ladder is not obtainable for rank-1 without "
    "repairing that line.",

    "INSTRUMENT B (the official evaluator) HAS NO WARMUP -- one obligatory "
    "correctness call, then straight into the timed loop -- and its iteration "
    "count is adaptive (it stops at err/mean < 0.001, or 120 s, or max_repeats). "
    "It cannot produce medians, raw samples, a fixed 3x50 protocol, a pipelined "
    "region, or a null arm, which is why it is the cross-check and Instrument A "
    "is the ladder.",
]


def geomean(values):
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ------------------------------------------------------------ popcorn parsing ---

SPEC_RE = re.compile(r"m:\s*(\d+);\s*n:\s*(\d+);\s*k:\s*(\d+)")


class PopcornError(RuntimeError):
    pass


def parse_popcorn(path, expect=6):
    """Parse one `benchmark.<i>.<field>` popcorn file. Times are NANOSECONDS.

    Fails loudly on: no count line, a count other than `expect`, a missing block,
    a missing best/mean, a `status: fail`, or a spec that does not match the
    canonical graded shape at that index. Silent nulls are the failure mode this
    parser exists to prevent.
    """
    if not os.path.exists(path):
        raise PopcornError(f"{path}: does not exist")
    text = open(path, errors="replace").read()
    if not text.strip():
        raise PopcornError(f"{path}: empty")

    count = None
    fields = {}
    check = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("benchmark-count:"):
            count = int(line.split(":", 1)[1])
            continue
        if line.startswith("check:"):
            check = line.split(":", 1)[1].strip()
            continue
        match = re.match(r"^benchmark\.(\d+)\.([a-z_]+):\s*(.*)$", line)
        if match:
            index, key, value = int(match.group(1)), match.group(2), match.group(3)
            fields.setdefault(index, {})[key] = value

    if count is None:
        raise PopcornError(f"{path}: no `benchmark-count:` line -- this is not a "
                           f"benchmark popcorn file (test runs emit `test-count:`)")
    if count != expect:
        raise PopcornError(f"{path}: benchmark-count is {count}, expected {expect}")

    per_shape, missing = [], []
    for index in range(expect):
        block = fields.get(index)
        if not block:
            missing.append(index)
            continue
        if block.get("status") == "fail":
            raise PopcornError(f"{path}: shape {index} reported status=fail: "
                               f"{block.get('error', '(no error text)')}")
        if "best" not in block or "mean" not in block:
            missing.append(index)
            continue
        spec = block.get("spec", "")
        found = SPEC_RE.search(spec)
        label = f"{found.group(1)}x{found.group(2)}x{found.group(3)}" if found else None
        if label != SHAPE_LABELS[index]:
            raise PopcornError(f"{path}: shape {index} spec is {label!r}, expected "
                               f"{SHAPE_LABELS[index]!r} -- the cases file is not "
                               f"the six graded shapes in order")
        per_shape.append({
            "shape": label,
            # eval.py's calculate_stats works in nanoseconds; everything this
            # project reports is microseconds.
            "best_us": float(block["best"]) / 1000.0,
            "mean_us": float(block["mean"]) / 1000.0,
            "median_us": None,      # eval.py does not emit a median
            "worst_us": float(block["worst"]) / 1000.0 if "worst" in block else None,
            "std_us": float(block["std"]) / 1000.0 if "std" in block else None,
            "rsd_pct": (float(block["std"]) / float(block["mean"]) * 100.0
                        if block.get("mean") and float(block["mean"]) > 0 else None),
            "runs": int(block["runs"]) if "runs" in block else None,
            "samples_us": [],       # eval.py does not emit raw samples
        })

    if missing:
        raise PopcornError(
            f"{path}: benchmark-count says {expect} but shapes {missing} have no "
            f"usable block. This is the truncated-run hazard (an evaluator that "
            f"hit its timeout wall still emits the count header); refusing to "
            f"emit a partial ladder.")

    return {
        "source": path,
        "check": check,
        "per_shape": per_shape,
        "geomean_us": geomean([s["best_us"] for s in per_shape]),
        "geomean_mean_us": geomean([s["mean_us"] for s in per_shape]),
        "statistic_note": "geomean_us is over per-shape BEST; eval.py emits no median",
    }


# --------------------------------------------------- instrument A aggregation ---

def load_ladder(root):
    """Pool the eight ranks of every shape's ladder_mp.py output."""
    per_shape_raw = {}
    for path in sorted(glob.glob(os.path.join(root, "raw", "ladder",
                                              "lad_s*.rank*.json"))):
        base = os.path.basename(path)
        try:
            index = int(base.split(".")[0].replace("lad_s", ""))
            rank = int(base.split(".")[1].replace("rank", ""))
        except ValueError:
            continue
        try:
            data = json.load(open(path))
        except Exception as exc:
            print(f"WARNING: {path} unreadable ({exc})", file=sys.stderr)
            continue
        if data:
            per_shape_raw.setdefault(index, {})[rank] = data
    return per_shape_raw


def reduce_cell(samples):
    if not samples:
        return None
    return {
        "best_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "worst_us": max(samples),
        "rsd_pct": (statistics.stdev(samples) / statistics.mean(samples) * 100.0
                    if len(samples) > 1 else 0.0),
        "n": len(samples),
        "samples_us": samples,
    }


def build_protocols(per_shape_raw, strict, expect=6):
    protocols = {}
    correctness = {}
    for proto in PROTOCOLS:
        arms = {}
        for arm in ALL_ARMS:
            rows, rank0_rows = [], []
            for index in range(expect):
                shaped = per_shape_raw.get(index)
                if not shaped:
                    rows.append(None)
                    continue
                pooled, rank0 = [], []
                label = None
                for rank in sorted(shaped):
                    entry = shaped[rank]
                    label = entry.get("shape_label")
                    got = entry.get("arms", {}).get(arm, {}).get(proto, [])
                    pooled.extend(got)
                    if rank == 0:
                        rank0.extend(got)
                    for key, value in entry.get("correctness", {}).items():
                        correctness.setdefault(label, {}).setdefault(key, value)
                cell = reduce_cell(pooled)
                if cell is None:
                    rows.append(None)
                    continue
                cell["shape"] = label or SHAPE_LABELS[index]
                cell["rank0_us"] = reduce_cell(rank0) and {
                    "best_us": min(rank0), "median_us": statistics.median(rank0),
                    "mean_us": statistics.mean(rank0), "n": len(rank0),
                }
                rows.append(cell)
                rank0_rows.append(cell["rank0_us"])
            present = [r for r in rows if r]
            if not present:
                continue
            if strict and len(present) != expect:
                raise RuntimeError(
                    f"instrument A: arm {arm!r} protocol {proto!r} has "
                    f"{len(present)} of {expect} shapes. Refusing to emit a "
                    f"partial ladder; re-run the missing shapes or pass "
                    f"--lenient and say so in result.md.")
            arms[arm] = {
                "per_shape": rows,
                "geomean_us": geomean([r["best_us"] for r in present]),
                "geomean_median_us": geomean([r["median_us"] for r in present]),
                "geomean_mean_us": geomean([r["mean_us"] for r in present]),
                "shapes_present": len(present),
            }
        if arms:
            protocols[proto] = {"arms": arms}
    return protocols, correctness


def net_out_floor(protocols):
    """Graded, with the measured harness constant removed from every arm.

    plan.md 7: the graded ladder is reported raw AND netted, never one alone.
    The quantity comes from this run's harness_floor arm rather than from the
    remembered 92 us, so if the floor turns out to scale with shape size the
    netting falsifies itself visibly instead of quietly.
    """
    graded = protocols.get("graded", {}).get("arms", {})
    floor = graded.get("harness_floor")
    if not floor:
        return None
    netted = {"floor_source": "harness_floor arm, measured in this run",
              "floor_per_shape_us": [], "arms": {}}
    floor_rows = floor["per_shape"]
    for row in floor_rows:
        netted["floor_per_shape_us"].append(row["median_us"] if row else None)
    netted["floor_is_shape_independent"] = None
    present = [v for v in netted["floor_per_shape_us"] if v]
    if len(present) > 1:
        spread = (max(present) - min(present)) / statistics.mean(present) * 100.0
        netted["floor_spread_pct"] = spread
        # plan.md 7 only holds if the floor is roughly flat across shapes.
        netted["floor_is_shape_independent"] = spread < 25.0
    for arm in RATIO_ARMS + ["ours_null"]:
        if arm not in graded:
            continue
        rows = []
        for row, sub in zip(graded[arm]["per_shape"], netted["floor_per_shape_us"]):
            if not row or not sub:
                rows.append(None)
                continue
            rows.append({
                "shape": row["shape"],
                "best_us": max(row["best_us"] - sub, 0.0),
                "median_us": max(row["median_us"] - sub, 0.0),
                "mean_us": max(row["mean_us"] - sub, 0.0),
            })
        present_rows = [r for r in rows if r]
        netted["arms"][arm] = {
            "per_shape": rows,
            "geomean_us": geomean([r["best_us"] for r in present_rows]),
            "geomean_median_us": geomean([r["median_us"] for r in present_rows]),
        }
    return netted


def ratio_block(numer, denom, stat="best_us"):
    if not numer or not denom:
        return None
    per_shape = []
    for a, b in zip(numer["per_shape"], denom["per_shape"]):
        if not a or not b or not b.get(stat):
            per_shape.append(None)
            continue
        per_shape.append({"shape": a["shape"],
                          "ratio": a[stat] / b[stat],
                          "ours_us": a[stat], "other_us": b[stat]})
    present = [r["ratio"] for r in per_shape if r]
    return {"statistic": stat, "per_shape": per_shape, "geomean": geomean(present)}


def build_ratios(protocols, netted):
    out = {}
    for proto, block in protocols.items():
        arms = block["arms"]
        for other, name in (("rank1", "ours_vs_rank1"),
                            ("reference", "ours_vs_reference")):
            for stat in ("best_us", "median_us"):
                key = f"{name}__{proto}__{stat.replace('_us', '')}"
                got = ratio_block(arms.get("ours"), arms.get(other), stat)
                if got:
                    out[key] = got
        # The null arm's own ratio IS the ladder's noise floor, in the same run.
        got = ratio_block(arms.get("ours"), arms.get("ours_null"), "best_us")
        if got:
            out[f"null_floor__{proto}__best"] = got
    if netted:
        for other, name in (("rank1", "ours_vs_rank1"),
                            ("reference", "ours_vs_reference")):
            got = ratio_block(netted["arms"].get("ours"),
                              netted["arms"].get(other), "median_us")
            if got:
                out[f"{name}__graded_netted__median"] = got
    # Flat aliases matching the schema stated in result.md.
    if "ours_vs_rank1__graded__best" in out:
        out["ours_vs_rank1"] = out["ours_vs_rank1__graded__best"]
    if "ours_vs_reference__graded__best" in out:
        out["ours_vs_reference"] = out["ours_vs_reference__graded__best"]
    return out


# ------------------------------------------------------- crosscheck assembly ---

def build_crosscheck(root, strict):
    """Instrument B: parse every captured popcorn benchmark run, per rotation."""
    out, errors = {}, []
    pattern = os.path.join(root, "raw", "eval", "rot*", "*", "*.popcorn.txt")
    for path in sorted(glob.glob(pattern)):
        parts = path.replace("\\", "/").split("/")
        rot, arm_dir = parts[-3], parts[-2]
        name = os.path.basename(path)
        if not name.startswith("benchmark"):
            continue           # test.popcorn.txt has no timing blocks
        try:
            out.setdefault(rot, {})[arm_dir] = parse_popcorn(path)
        except PopcornError as exc:
            errors.append(str(exc))
    if errors and strict:
        raise RuntimeError("instrument B parse failures (pass --lenient to record "
                           "them instead of raising):\n  " + "\n  ".join(errors))
    return {"rotations": out, "parse_errors": errors}


def node_block():
    def read(path):
        try:
            return open(path, errors="replace").read()
        except Exception:
            return None
    return {
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "arch": "gfx942 MI300X SPX, 304 CU/GPU, 8 devices",
        "container": "dhk-gemmrs (ours + reference + ladder), uid 15523",
        "provenance": read(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "logs", "provenance.txt")),
    }


# ---------------------------------------------------------------------- main ---

def print_fixture(paths):
    """Validate the popcorn parser against saved historical output. No GPU."""
    if not paths:
        paths = [f"{ON}/compbench/{arm}/benchmark.popcorn.txt"
                 for arm in ("ours", "reference", "rank1")]
    rc = 0
    for path in paths:
        arm = os.path.basename(os.path.dirname(path))
        print(f"\n=== fixture: {arm}  ({path}) ===")
        try:
            parsed = parse_popcorn(path)
        except PopcornError as exc:
            print(f"  PopcornError (this is the parser working, if the file is "
                  f"known-truncated): {exc}")
            rc = max(rc, 1)
            continue
        print(f"  {'#':>2} {'shape':>21} {'best us':>10} {'mean us':>10} "
              f"{'worst us':>11} {'rsd%':>7} {'runs':>5}")
        for i, row in enumerate(parsed["per_shape"]):
            print(f"  {i + 1:>2} {row['shape']:>21} {row['best_us']:>10.2f} "
                  f"{row['mean_us']:>10.2f} {row['worst_us']:>11.2f} "
                  f"{row['rsd_pct']:>7.1f} {row['runs']:>5}")
        print(f"  geomean(best) = {parsed['geomean_us']:.2f} us   "
              f"geomean(mean) = {parsed['geomean_mean_us']:.2f} us   "
              f"check = {parsed['check']}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--lenient", action="store_true",
                    help="record missing shapes instead of raising")
    ap.add_argument("--fixture", nargs="*", default=None,
                    help="parse saved popcorn output and print it; no GPU")
    args = ap.parse_args()

    if args.fixture is not None:
        return print_fixture(args.fixture)

    strict = not args.lenient
    per_shape_raw = load_ladder(args.root)
    if not per_shape_raw:
        print(f"no instrument-A output under {args.root}/raw/ladder", file=sys.stderr)
        if strict:
            return 1
    protocols, correctness = build_protocols(per_shape_raw, strict)
    netted = net_out_floor(protocols)
    ratios = build_ratios(protocols, netted)
    crosscheck = build_crosscheck(args.root, strict)

    doc = {
        "experiment": "exp_24_ladders",
        "node": node_block(),
        "config": {
            "shapes": SHAPE_LABELS,
            "arms": ALL_ARMS,
            "ratio_arms": RATIO_ARMS,
            "null_arms": NULL_ARMS,
            "protocols": PROTOCOLS,
            "instrument_a": "ladder_mp.py -- five arms, one 8-process pool per "
                            "shape, arm order cyclically rotated so every arm is "
                            "first exactly once, protocol order flipped per rep, "
                            "duration-based warmup, all 8 ranks pooled",
            "instrument_b": "the official evaluator via tools/run_ours_evaluator.sh, "
                            "tools/run_reference_arm.sh and "
                            "experiments/exp_10_rank1/r1_eval.sh, arm order rotated",
            "bias_forced_all_shapes": True,
        },
        "protocols": protocols,
        "graded_netted": netted,
        "correctness": correctness,
        "ratios": ratios,
        "evaluator_crosscheck": crosscheck,
        "prior_denominator": {
            "source": "exp_14 result.md section 8, same-run graded, best-of-arm",
            "rank1_geomean_us": 314.46,
            "ours_geomean_us": 345.15,
            "reference_geomean_us": 438.15,
            "ours_vs_rank1_geomean": 1.098,
            "ours_vs_rank1_per_shape": [0.850, 1.072, 1.102, 1.120, 1.286, 1.208],
            "ratio_noise_floor_pct": 2.0,
        },
        "caveats": CAVEATS,
    }

    out = args.out or os.path.join(args.root, "ladders.json")
    with open(out, "w") as handle:
        json.dump(doc, handle, indent=2)
    print(f"wrote {out}  ({os.path.getsize(out)} bytes)")

    for proto, block in protocols.items():
        print(f"\n--- {proto} (geomean of per-shape BEST) ---")
        for arm, entry in block["arms"].items():
            print(f"  {arm:>14}  {entry['geomean_us']:9.2f} us  "
                  f"(median {entry['geomean_median_us']:9.2f})")
    for key, entry in ratios.items():
        if entry and entry.get("geomean"):
            print(f"  ratio {key:<44} {entry['geomean']:.4f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
