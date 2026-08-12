#!/usr/bin/env python3
"""exp_25 — per-shape sensitivity readout (paper Q5).

Derives `knob_by_shape.json` + `sensitivity_points.csv` from two artifacts that
already exist on disk:

  aug11/exp_20_attribution/ablation.json   stage attribution + geometry + floors
  aug11/exp_23_waterfall/waterfall.json    per-shape rung ladder + NR sweep
  aug11/exp_23_waterfall/stats.json        scored contrasts + widened null floors

No GPU, no measurement, no network. Every number in the output is derived here;
none is typed by hand. Run with --check to re-verify the derived geometry and
the structural-activity flags against the inputs independently.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUG11 = os.path.dirname(HERE)
ABL_PATH = os.path.join(AUG11, "exp_20_attribution", "ablation.json")
WF_PATH = os.path.join(AUG11, "exp_23_waterfall", "waterfall.json")
ST_PATH = os.path.join(AUG11, "exp_23_waterfall", "stats.json")

GRID_CTAS = 304          # MI300X SPX, hardcoded dim3(m3::CU_COUNT) in the kernel
LDS_CAP_BYTES = 65536    # 64 KB per workgroup on gfx942
RELEASE_GROUP = 4        # shipped HK_GEMM_RS_MI300X_RELEASE_GROUP
WORLD = 8

# Shapes excluded from every comm-share number and every correlation. Reason is
# fixed in plan.md §3 and is a property of exp_20's own reading, not of the
# numbers this script computes.
EXCLUDED = {
    1: ("HOST-bound: bound=HOST at 10.39x SOL (62.34 us of issue inside a "
        "66.41 us wall); 3 of 5 stage deltas negative; all 5 inside the 2.34 us "
        "floor; the five cuts price only 4.9% of wall time"),
    2: ("HOST-bound against its median (62.43 us of issue inside a 67.80 us "
        "median wall); 1 of 5 stage deltas negative; the five cuts price only "
        "9.5% of wall time"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- statistics --
def ranks(xs):
    """Average ranks, 1 = smallest."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def contrast(stats_shape, prefix):
    """exp_23 keys a contrast 'nr32 vs c' or 'nr32 vs c *shipped'."""
    for key, val in stats_shape["contrasts"].items():
        if key == prefix or key.startswith(prefix + " "):
            return key, val
    return None, None


# ------------------------------------------------------------------- derived --
def derive_geometry(s):
    g = s["geometry"]
    bm, bn, bk = s["bm"], s["bn"], s["bk"]
    nr = s["nr"]
    ng = GRID_CTAS - nr
    tiles = ((s["m"] + bm - 1) // bm) * ((s["n"] + bn - 1) // bn)
    tiles_per_cta = (tiles + ng - 1) // ng
    lds = 2 * (bm + bn) * bk * 2          # double-buffered A+B tiles, bf16
    k_local = s["k"] // WORLD
    k_iters = (k_local + bk - 1) // bk
    return {
        "bm": bm, "bn": bn, "bk": bk,
        "num_reducer_ctas": nr,
        "num_gemm_ctas": ng,
        "grid_ctas": GRID_CTAS,
        "tiles": tiles,
        "tiles_per_cta": tiles_per_cta,
        "producer_waves": g["producer_waves"],
        "last_wave_fill": g["last_wave_fill"],
        "k_local": k_local,
        "k_iters": k_iters,
        "waves_x_k_iters": g["waves_x_k_iters"],
        "lds_double_buffer_bytes": lds,
        "at_64kb_lds_cap": lds >= LDS_CAP_BYTES,
        "effective_release_group": g["effective_release_group"],
    }


def derive_arithmetic(s, geo):
    m, n, k = s["m"], s["n"], s["k"]
    flops = 2.0 * m * n * k
    # a rank produces the full MxN output slab and sends 7 of 8 slices off-rank
    egress_bytes = (WORLD - 1) / WORLD * m * n * 2
    return {
        "flops": flops,
        "egress_bytes_mb": egress_bytes / 1e6,
        "flops_per_egress_byte": flops / egress_bytes,
        "flops_per_egress_byte_closed_form": 8.0 * k / 7.0,
        "note_closed_form": (
            "flops/egress byte = 2MNK / ((7/8) MN 2) = 8K/7 exactly, so the "
            "compute-per-communicated-byte axis of the graded set is a function "
            "of K alone. M and N enter only through the tile count."
        ),
        "tiles_over_ng": geo["tiles"] / geo["num_gemm_ctas"],
    }


def comm_shares(s):
    st = s["stages_us"]
    full = s["full_us"]
    ssum = sum(st.values())
    comm = st["egress"] + st["sync"] + st["release"]
    return {
        "D1_egress_over_full": st["egress"] / full,
        "D2_egress_sync_release_over_full": comm / full,
        "D3_full_minus_gemm_over_full": (full - st["gemm"]) / full,
        "D4_comm_over_priced_pools": comm / ssum,
        "stage_sum_over_full": ssum / full,
    }


DEFS = [
    "D1_egress_over_full",
    "D2_egress_sync_release_over_full",
    "D3_full_minus_gemm_over_full",
    "D4_comm_over_priced_pools",
]
PRIMARY_DEF = "D3_full_minus_gemm_over_full"


def build():
    abl = load(ABL_PATH)
    wf = load(WF_PATH)
    st = load(ST_PATH)

    rung_by_name = {r["rung"]: r for r in wf["rungs"]}
    nr_by_n = {a["nr"]: a for a in wf["nr_sweep"]}

    shapes_out = []
    for s in abl["shapes"]:
        idx = s["shape_index"]
        sst = st["shapes"][str(idx)]
        geo = derive_geometry(s)
        arith = derive_arithmetic(s, geo)
        shares = comm_shares(s)
        floor_pct = sst["null_floor_worst_draw_pct"]

        # ---- structural activity, computed from the shape plan, not measured --
        order_active = geo["tiles"] > geo["num_gemm_ctas"]
        gran_active = geo["tiles_per_cta"] >= RELEASE_GROUP

        # ---- rung deltas ---------------------------------------------------
        med = {r: rung_by_name[r]["per_shape"][idx - 1]["median_us"]
               for r in ("a", "b", "c", "null")}
        best = {r: rung_by_name[r]["per_shape"][idx - 1]["best_us"]
                for r in ("a", "b", "c", "null")}

        rungs = {}
        for name, prefix, prev, cur, active, mech in (
            ("a_to_b", "b vs a", "a", "b", order_active,
             "WGM destination-spreading task order; differs from the constant "
             "WGM=4 only where tiles > num_gemm_ctas"),
            ("b_to_c", "c vs b", "b", "c", gran_active,
             "grouped release, RELEASE_GROUP=4 with FULL_ONLY=1; rgroup "
             "collapses to 1 unless tiles_per_cta >= 4"),
        ):
            _, c = contrast(sst, prefix)
            rungs[name] = {
                "mechanism": mech,
                "structurally_active": active,
                "gain_median_pct": c["gain_median_pct"],
                "gain_best_pct": c["gain_best_pct"],
                "gain_median_per_draw_range_pct": c["gain_median_range"],
                "delta_us_from_pooled_medians": med[prev] - med[cur],
                "delta_us_from_pooled_bests": best[prev] - best[cur],
                "floor_pct": floor_pct,
                "floor_us": floor_pct / 100.0 * med[cur],
                "beyond_floor": c["beyond_floor"],
                "verdict": c["verdict"],
                "resolved": c["verdict"].startswith("RESOLVED"),
                "note_delta_us": (
                    "the percentages are exp_23's within-draw paired contrasts; "
                    "delta_us is recomputed from the pooled medians/bests and is "
                    "therefore only approximately their product with the wall"
                ),
            }

        # ---- rung delta against the pool it nominally targets --------------
        # The comm pool is measured at the TOP of the ladder (rung c) while the
        # delta is taken from the rung below it, so a ratio above 1 is not a
        # contradiction: it is proof that the knob shrank the pool it acted on.
        stg = s["stages_us"]
        comm_pool = stg["egress"] + stg["sync"] + stg["release"]
        for name, target in (("a_to_b", "egress"), ("b_to_c", "release")):
            d_us = rungs[name]["delta_us_from_pooled_medians"]
            rungs[name]["vs_pools_at_rung_c"] = {
                "comm_pool_us_D2_numerator": comm_pool,
                "target_pool": target,
                "target_pool_us": stg[target],
                "target_pool_clears_exp20_floor":
                    s["stage_clears_floor"][target],
                "delta_over_comm_pool": d_us / comm_pool if comm_pool else None,
                "delta_over_target_pool":
                    d_us / stg[target] if stg[target] else None,
                "delta_over_full_at_rung_c": d_us / med["c"],
                "note": ("the pools are exp_20's single-cut deltas at rung c; a "
                         "ratio > 1 means the rung's win exceeds the pool that "
                         "survives at the winner, i.e. the knob removed the pool "
                         "it would otherwise be regressed against"),
            }

        # ---- NR curve ------------------------------------------------------
        shipped_nr = sst["shipped_nr"]
        nr_points = []
        for n_red in sorted(nr_by_n):
            key, c = contrast(sst, "nr%d vs c" % n_red)
            row = nr_by_n[n_red]["per_shape"][idx - 1]
            nr_points.append({
                "nr": n_red,
                "num_gemm_ctas": GRID_CTAS - n_red,
                "is_shipped_config": n_red == shipped_nr,
                "median_us": row["median_us"],
                "best_us": row["best_us"],
                "gain_median_pct_vs_shipped": c["gain_median_pct"],
                "gain_best_pct_vs_shipped": c["gain_best_pct"],
                "within_widened_floor": abs(c["gain_median_pct"]) < floor_pct,
                "beyond_floor": c["beyond_floor"],
                "verdict": c["verdict"],
                "is_null_pair": c["is_null_pair"],
                "contrast_key": key,
                # exp_23's disjointness rule can certify a contrast RESOLVED
                # while the same contrast sits inside the widened union-of-pairs
                # floor. Where that happens it is flagged, and this readout
                # treats the floor as authoritative.
                "verdict_conflicts_with_widened_floor": (
                    c["verdict"].startswith("RESOLVED")
                    and abs(c["gain_median_pct"]) < floor_pct),
            })
        plateau = [p for p in nr_points if p["nr"] >= 32]
        cliff = [p for p in nr_points if p["nr"] < 32]
        # lower edge of the flat region: the smallest swept NR from which every
        # larger swept NR is inside this shape's widened floor
        edge = None
        for p in sorted(nr_points, key=lambda q: q["nr"]):
            if all(q["within_widened_floor"]
                   for q in nr_points if q["nr"] >= p["nr"]):
                edge = p["nr"]
                break
        nr_curve = {
            "shipped_nr": shipped_nr,
            "reference_arm": "rung c at this shape's shipped NR",
            "points": nr_points,
            "plateau_points_nr_ge_32": [p["nr"] for p in plateau],
            "flat_from_nr32_upward": all(
                p["within_widened_floor"] for p in plateau),
            "plateau_lower_edge_nr": edge,
            "plateau_max_abs_gain_pct": max(
                abs(p["gain_median_pct_vs_shipped"]) for p in plateau),
            "points_within_widened_floor": [
                p["nr"] for p in nr_points if p["within_widened_floor"]],
            "points_resolved_slower": [
                p["nr"] for p in nr_points
                if p["verdict"].startswith("RESOLVED slower")],
            "cliff_points_nr_lt_32": [p["nr"] for p in cliff],
            "cliff_all_resolved_slower": all(
                p["verdict"].startswith("RESOLVED slower") for p in cliff),
            "cliff_worst_gain_pct": min(
                p["gain_median_pct_vs_shipped"] for p in cliff),
        }

        usable = idx not in EXCLUDED
        shapes_out.append({
            "shape_index": idx,
            "shape": s["shape"],
            "m": s["m"], "n": s["n"], "k": s["k"], "has_bias": s["has_bias"],
            "geometry": geo,
            "derived_arithmetic": arith,
            "attribution": {
                "full_us": s["full_us"],
                "stages_us": s["stages_us"],
                "stage_share_of_full": s["stage_share"],
                "stage_clears_exp20_floor": s["stage_clears_floor"],
                "stage_sum_over_full": s["stage_sum_over_full"],
                "x_sol": s["x_sol"],
                "m7_bound": s["m7"]["bound"],
                "m7_host_issue_us_per_op": s["m7"]["host_issue_us_per_op"],
            },
            "comm_share": {
                "usable_for_correlation": usable,
                "exclusion_reason": EXCLUDED.get(idx),
                **({k: v for k, v in shares.items()} if usable else
                   {k: None for k in DEFS} | {
                       "stage_sum_over_full": shares["stage_sum_over_full"]}),
            },
            "floors": {
                "exp20_allocation_floor_pct": s["floor_pct"],
                "exp20_allocation_floor_us": s["floor_us"],
                "exp23_widened_floor_pct": floor_pct,
                "exp23_widened_floor_us": floor_pct / 100.0 * med["c"],
                "exp23_single_pair_floor_pct": sst["null_floor_single_pair_pct"],
                "exp23_null_pairs": sst["null_pairs"],
                "which_is_used": "exp23_widened_floor_pct for every rung and NR "
                                 "verdict; it is measured in the run that "
                                 "produced those ratios, over the union of all "
                                 "identically-configured pairs",
            },
            "arm_medians_us": med,
            "arm_bests_us": best,
            "rungs": rungs,
            "nr_curve": nr_curve,
            "knob_structurally_active": {
                "order_a_to_b": order_active,
                "granularity_b_to_c": gran_active,
                "nr_reduction_below_plateau": True,
            },
        })

    # ------------------------------------------------------------ correlations --
    usable = [s for s in shapes_out if s["comm_share"]["usable_for_correlation"]]
    corr = {
        "n_shapes_entering": len(usable),
        "shapes_entering": [s["shape_index"] for s in usable],
        "shapes_excluded": {str(k): v for k, v in EXCLUDED.items()},
        "min_attainable_two_sided_p_at_this_n": 2.0 / math.factorial(len(usable)),
        "why_no_fit": (
            "n=4. A perfect rank reversal cannot reach p<0.05 at n=4 "
            "(2/4! = 0.083), so no coefficient here is inferential and no line "
            "is fitted. Ordering statements only."
        ),
        "by_definition": {},
        "active_only": {},
        "collinearity": {},
    }

    for d in DEFS:
        xs = [s["comm_share"][d] for s in usable]
        for rung in ("a_to_b", "b_to_c"):
            ys = [s["rungs"][rung]["gain_median_pct"] for s in usable]
            corr["by_definition"].setdefault(d, {})[rung + "_mask_variant"] = {
                "spearman_rho": spearman(xs, ys),
                "pearson_r": pearson(xs, ys),
                "n": len(xs),
                "points": [
                    {"shape_index": s["shape_index"],
                     "comm_share": s["comm_share"][d],
                     "gain_median_pct": s["rungs"][rung]["gain_median_pct"],
                     "structurally_active": s["rungs"][rung]["structurally_active"],
                     "resolved": s["rungs"][rung]["resolved"]}
                    for s in usable],
                "interpretation": (
                    "MASK VARIANT: includes shapes where the knob is "
                    "arithmetically inert, so this measures the structural mask, "
                    "not sensitivity. Unresolved entries are noise, not zeros."
                ),
            }
        # ordering of shapes by this definition, most-comm-first
        order = sorted(usable, key=lambda s: -s["comm_share"][d])
        corr["by_definition"][d]["shape_order_most_comm_first"] = [
            s["shape_index"] for s in order]
        corr["by_definition"][d]["order_active_shapes_are_the_two_lowest_comm"] = (
            {s["shape_index"] for s in order[-2:]}
            == {s["shape_index"] for s in usable
                if s["rungs"]["a_to_b"]["structurally_active"]})
        corr["by_definition"][d]["comm_share_rank_of_order_active_shapes"] = {
            str(s["shape_index"]): order.index(s) + 1
            for s in usable if s["rungs"]["a_to_b"]["structurally_active"]}

    for rung, label in (("a_to_b", "order"), ("b_to_c", "granularity")):
        act = [s for s in usable if s["rungs"][rung]["structurally_active"]]
        corr["active_only"][rung] = {
            "knob": label,
            "n_active_and_usable": len(act),
            "coefficient": None,
            "why_none": (
                "n=%d. Two points define a slope by construction and one point "
                "defines nothing. No correlation is computed." % len(act)),
            "points": [
                {"shape_index": s["shape_index"],
                 "shape": s["shape"],
                 **{d: s["comm_share"][d] for d in DEFS},
                 "gain_median_pct": s["rungs"][rung]["gain_median_pct"],
                 "delta_us": s["rungs"][rung]["delta_us_from_pooled_medians"],
                 "tiles_over_ng": s["derived_arithmetic"]["tiles_over_ng"],
                 "tiles_per_cta": s["geometry"]["tiles_per_cta"],
                 "k": s["k"]}
                for s in act],
        }

    # Alternative axis for the NR cliff: it should track the OWNER-SIDE REDUCE
    # share, not the communication share, if the reducers are a computation pool
    # rather than a communication pool. n = 4, so this is a direction, not a fit.
    def nr8(s):
        return next(p["gain_median_pct_vs_shipped"]
                    for p in s["nr_curve"]["points"] if p["nr"] == 8)

    red = [s["attribution"]["stage_share_of_full"]["reduce"] for s in usable]
    pen = [abs(nr8(s)) for s in usable]
    corr["nr_cliff_axis"] = {
        "y": "absolute NR=8 penalty, % vs that shape's shipped NR",
        "n": len(usable),
        "spearman_vs_reduce_share": spearman(red, pen),
        "spearman_vs_" + PRIMARY_DEF: spearman(
            [s["comm_share"][PRIMARY_DEF] for s in usable], pen),
        "spearman_vs_D1_egress_over_full": spearman(
            [s["comm_share"]["D1_egress_over_full"] for s in usable], pen),
        "points": [{"shape_index": s["shape_index"],
                    "reduce_share": s["attribution"]["stage_share_of_full"]["reduce"],
                    "nr8_penalty_pct": nr8(s)} for s in usable],
        "interpretation": (
            "Reported so the NR cliff is not silently attributed to the "
            "communication share. It correlates with the reduce share (+0.60) "
            "and with the not-GEMM share (+0.80) and NOT AT ALL with egress/full "
            "(0.00) - but at n=4 these three are statistically "
            "indistinguishable and NONE of them is a claim. The load-bearing "
            "argument for the cliff is mechanical, not correlational: the "
            "reducers run the owner-side reduce, and NR=8 makes shape 6's reduce "
            "take 16 rounds instead of 3."
        ),
    }

    ks = [float(s["k"]) for s in usable]
    tn = [s["derived_arithmetic"]["tiles_over_ng"] for s in usable]
    corr["collinearity"] = {
        "spearman_k_vs_tiles_over_ng": spearman(ks, tn),
        "spearman_k_vs_" + PRIMARY_DEF: spearman(
            ks, [s["comm_share"][PRIMARY_DEF] for s in usable]),
        "shapes": [s["shape_index"] for s in usable],
        "k_values": [s["k"] for s in usable],
        "tiles_over_ng_values": tn,
        "interpretation": (
            "Comm share is a function of K (8K/7 flops per egress byte); knob "
            "activity is a function of tiles/NG, i.e. of M*N over the tile area "
            "and the producer count. Across the four usable graded shapes these "
            "two axes are rank-identical, so the structural mask and the comm "
            "share are perfectly confounded and cannot be separated by this "
            "shape set at any n."
        ),
    }

    out = {
        "experiment": "aug11/exp_25_sensitivity",
        "what": ("per-shape sensitivity readout (paper Q5): each waterfall "
                 "rung's delta per shape, beside that shape's measured "
                 "communication share and its own measured noise floor, with "
                 "structural activity separated from sensitivity"),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_runs": 0,
        "derivation": "pure analysis over exp_20 and exp_23; no new measurement",
        "inputs": {
            "ablation": {"path": "aug11/exp_20_attribution/ablation.json",
                         "sha256": sha256(ABL_PATH)},
            "waterfall": {"path": "aug11/exp_23_waterfall/waterfall.json",
                          "sha256": sha256(WF_PATH)},
            "stats": {"path": "aug11/exp_23_waterfall/stats.json",
                      "sha256": sha256(ST_PATH)},
        },
        "protocol": ("pipelined, one process driving 8 devices, which is NOT the "
                     "official evaluator's topology; exp_20 at 40 iters/cell, "
                     "exp_23 at 4 allocation draws x 8-arm rotation, best+median "
                     "only (means are unusable on this node)"),
        "comm_share_definitions": {
            "D1_egress_over_full": "stages_us.egress / full_us",
            "D2_egress_sync_release_over_full":
                "(egress + sync + release) / full_us",
            "D3_full_minus_gemm_over_full": "(full_us - stages_us.gemm) / full_us",
            "D4_comm_over_priced_pools":
                "(egress + sync + release) / sum(all five stage deltas)",
            "why_four": ("exp_20's cuts are single-cut deltas: they OVERLAP and "
                         "do not partition full_us (stage_sum_over_full runs "
                         "0.05-1.02), so egress/full is not a clean share and no "
                         "single definition is canonical. All four are reported."),
            "primary": PRIMARY_DEF,
            "why_primary": ("numerator and denominator come from the same single "
                            "cut, so it cannot be inflated by double-counting "
                            "overlapping pools, and it is monotone in the "
                            "shape's own arithmetic (8K/7 flops per egress byte)"),
        },
        "caveats": [
            "Single-cut stage deltas overlap and are NOT a budget; a share built "
            "from them is a ratio of overlapping quantities, never a partition.",
            "Shapes 1 and 2 are HOST-bound; no comm share is computed for them "
            "and they enter no correlation. n = 4.",
            "A rung delta of ~0 on a structurally inert shape is arithmetic, not "
            "a sensitivity measurement.",
            "Rung deltas are measured at the configuration the rung starts from, "
            "while the comm share is measured at the LADDER'S TOP (rung c). The "
            "knob changes the very share it would be regressed against, so the "
            "regression in P1 is ill-posed as well as underpowered.",
            "exp_23 proved one null twin under-measures the floor: two "
            "identically-configured shape-6 arms differed by up to 4.44% "
            "consistently in both construction orders. Every floor here is the "
            "widened union-of-pairs floor.",
        ],
        "shapes": shapes_out,
        "correlations": corr,
    }
    return out


def write_csv(out, path):
    cols = ["shape_index", "shape", "m", "n", "k", "tiles", "num_gemm_ctas",
            "tiles_per_cta", "tiles_over_ng", "waves_x_k_iters", "bm", "bn",
            "bk", "at_64kb_lds_cap", "shipped_nr", "full_us",
            "flops_per_egress_byte", "usable_for_correlation",
            "D1_egress_over_full", "D2_egress_sync_release_over_full",
            "D3_full_minus_gemm_over_full", "D4_comm_over_priced_pools",
            "widened_floor_pct", "a_to_b_active", "a_to_b_gain_median_pct",
            "a_to_b_delta_us", "a_to_b_verdict", "b_to_c_active",
            "b_to_c_gain_median_pct", "b_to_c_delta_us", "b_to_c_verdict",
            "nr8_gain_pct", "nr16_gain_pct", "nr32_gain_pct", "nr48_gain_pct",
            "nr_plateau_flat"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in out["shapes"]:
            g, a, cs, fl = (s["geometry"], s["derived_arithmetic"],
                            s["comm_share"], s["floors"])
            nr = {p["nr"]: p for p in s["nr_curve"]["points"]}
            w.writerow([
                s["shape_index"], s["shape"], s["m"], s["n"], s["k"],
                g["tiles"], g["num_gemm_ctas"], g["tiles_per_cta"],
                round(a["tiles_over_ng"], 4), g["waves_x_k_iters"],
                g["bm"], g["bn"], g["bk"], int(g["at_64kb_lds_cap"]),
                s["nr_curve"]["shipped_nr"], s["attribution"]["full_us"],
                round(a["flops_per_egress_byte"], 1),
                int(cs["usable_for_correlation"]),
                *[None if cs[d] is None else round(cs[d], 4) for d in DEFS],
                round(fl["exp23_widened_floor_pct"], 3),
                int(s["rungs"]["a_to_b"]["structurally_active"]),
                round(s["rungs"]["a_to_b"]["gain_median_pct"], 3),
                round(s["rungs"]["a_to_b"]["delta_us_from_pooled_medians"], 2),
                s["rungs"]["a_to_b"]["verdict"],
                int(s["rungs"]["b_to_c"]["structurally_active"]),
                round(s["rungs"]["b_to_c"]["gain_median_pct"], 3),
                round(s["rungs"]["b_to_c"]["delta_us_from_pooled_medians"], 2),
                s["rungs"]["b_to_c"]["verdict"],
                *[round(nr[n]["gain_median_pct_vs_shipped"], 3)
                  for n in (8, 16, 32, 48)],
                int(s["nr_curve"]["flat_from_nr32_upward"]),
            ])


def check(out):
    """Independent re-verification of everything this script derived."""
    abl = {s["shape_index"]: s for s in load(ABL_PATH)["shapes"]}
    st = load(ST_PATH)["shapes"]
    bad = []
    for s in out["shapes"]:
        i = s["shape_index"]
        g, ref = s["geometry"], abl[i]["geometry"]
        for key in ("tiles", "tiles_per_cta", "k_iters", "num_gemm_ctas",
                    "waves_x_k_iters", "effective_release_group"):
            mine = g[key] if key != "tiles" else g["tiles"]
            theirs = ref["gemm_tiles"] if key == "tiles" else ref[key]
            if mine != theirs:
                bad.append("shape %d %s: derived %s vs exp_20 %s"
                           % (i, key, mine, theirs))
        act = s["knob_structurally_active"]
        if act["order_a_to_b"] != st[str(i)]["ab_rung_active"]:
            bad.append("shape %d a->b activity disagrees with exp_23" % i)
        if act["granularity_b_to_c"] != st[str(i)]["bc_rung_active"]:
            bad.append("shape %d b->c activity disagrees with exp_23" % i)
        eff = RELEASE_GROUP if g["tiles_per_cta"] >= RELEASE_GROUP else 1
        if eff != ref["effective_release_group"]:
            bad.append("shape %d effective release group disagrees" % i)
    if bad:
        raise SystemExit("CHECK FAILED:\n  " + "\n  ".join(bad))
    print("CHECK PASSED: geometry, tiles_per_cta, waves x k_iters, effective "
          "release group and both structural-activity flags reproduce exp_20 "
          "and exp_23 independently for all 6 shapes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    out = build()
    jp = os.path.join(HERE, "knob_by_shape.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    cp = os.path.join(HERE, "sensitivity_points.csv")
    write_csv(out, cp)
    print("wrote %s (%d bytes)" % (os.path.basename(jp), os.path.getsize(jp)))
    print("wrote %s (%d bytes)" % (os.path.basename(cp), os.path.getsize(cp)))

    if args.check:
        check(out)

    # A short human summary so the run is self-describing.
    print("\ncomm share (usable shapes only):")
    print("  # |    D1 |    D2 |    D3 |    D4 | a->b %  | b->c %  | floor%")
    for s in out["shapes"]:
        cs = s["comm_share"]
        if not cs["usable_for_correlation"]:
            print("  %d | EXCLUDED (host-bound)" % s["shape_index"])
            continue
        print("  %d | %5.3f | %5.3f | %5.3f | %5.3f | %+7.2f | %+7.2f | %5.2f"
              % (s["shape_index"], cs[DEFS[0]], cs[DEFS[1]], cs[DEFS[2]],
                 cs[DEFS[3]], s["rungs"]["a_to_b"]["gain_median_pct"],
                 s["rungs"]["b_to_c"]["gain_median_pct"],
                 s["floors"]["exp23_widened_floor_pct"]))
    print("\nmask-variant rank correlation of a->b gain vs comm share:")
    for d in DEFS:
        c = out["correlations"]["by_definition"][d]["a_to_b_mask_variant"]
        print("  %-34s rho = %+.3f  (n=%d, order most-comm-first %s)"
              % (d, c["spearman_rho"], c["n"],
                 out["correlations"]["by_definition"][d]
                 ["shape_order_most_comm_first"]))
    coll = out["correlations"]["collinearity"]
    print("\ncollinearity: spearman(K, tiles/NG) = %+.3f ; spearman(K, D3) = %+.3f"
          % (coll["spearman_k_vs_tiles_over_ng"],
             coll["spearman_k_vs_" + PRIMARY_DEF]))


if __name__ == "__main__":
    main()
