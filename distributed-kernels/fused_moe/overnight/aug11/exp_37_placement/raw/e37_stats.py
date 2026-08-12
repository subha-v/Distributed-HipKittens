#!/usr/bin/env python3
"""exp_37 statistics + placement.json builder.

  usage: e37_stats.py e37_raw.json OUTDIR COMMIT

CLUSTERING POLICY (the whole point of this script)
--------------------------------------------------
The independent unit is the CAMPAIGN, not the rank, and not the rotation.
  * A campaign's 8 ranks share ONE input draw and ONE collective, so rank-level
    samples are not independent. The harness has already collapsed them: each
    rotation reports ONE rank-max-aligned p50.
  * The 5 rotations inside a campaign share that campaign's build, container
    launch and routing draw, so they are correlated too. A sibling experiment
    got t = -5.01 unclustered vs t = -0.73 clustered on identical data.
So inference uses ONE number per campaign (median over its 5 rotations), paired
across the interleaved rounds. The rotation-level t is computed too, labelled
INVALID, only to show the inflation factor.
"""
import json
import math
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 19: 2.093, 20: 2.086, 30: 2.042, 38: 2.024}
CTRL = "mode12_C16_g353_fr16"
RATCHET_US = 6482.7          # standing ratchet, aug11 STATUS.md (C=16,g=353,mode12,fr16)


def t95(df):
    if df <= 0:
        return float("nan")
    return T95.get(df) or T95[min(T95, key=lambda k: abs(k - df))]


def mean(xs):
    return sum(xs) / len(xs)


def ci95(xs):
    n = len(xs)
    m = mean(xs)
    if n < 2:
        return {"n": n, "mean": m, "sd": None, "sem": None, "ci95": None,
                "t": None, "df": n - 1}
    sd = statistics.stdev(xs)
    sem = sd / math.sqrt(n)
    hw = t95(n - 1) * sem
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "half_width": hw,
            "ci95": [m - hw, m + hw], "t": (m / sem) if sem else None,
            "df": n - 1}


def sign_test_p(diffs):
    xs = [d for d in diffs if d != 0]
    n = len(xs)
    if n == 0:
        return None
    k = min(sum(1 for d in xs if d < 0), sum(1 for d in xs if d > 0))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ranksum_exact_p(a, b):
    """Exact two-sided Mann-Whitney/Wilcoxon rank-sum p (small samples)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0 or na + nb > 20:
        return None
    allv = sorted(a + b)
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks.setdefault(allv[k], r)
        i = j + 1
    obs = sum(ranks[v] for v in a)
    idx = list(range(na + nb))
    tot = 0
    hits = 0
    centre = na * (na + nb + 1) / 2.0
    for combo in combinations(idx, na):
        s = sum(ranks[allv[c]] for c in combo)
        tot += 1
        if abs(s - centre) >= abs(obs - centre) - 1e-9:
            hits += 1
    return hits / tot


def main():
    raw = json.loads(Path(sys.argv[1]).read_text())
    outdir = Path(sys.argv[2])
    commit = sys.argv[3] if len(sys.argv) > 3 else None
    outdir.mkdir(parents=True, exist_ok=True)

    camps = []
    for c in raw["campaigns"]:
        rb = c["cfg_readback"]
        mc = rb.get("mps_config")
        if mc is None:
            print(f"SKIP {c['tag']}#{c['idx']}: cfg readback inconsistent {rb['distinct']}")
            continue
        g = c["gates"]
        runs_mps = [r["arm_p50_us"]["mps_mega"] for r in c["runs"]]
        runs_prod = [r["arm_p50_us"]["production"] for r in c["runs"]]
        camps.append({
            "tag": c["tag"], "idx": c["idx"], "outdir": c["outdir"],
            "arm": f"mode{mc['mode']}_C{mc['C']}_g{mc['g']}_fr{mc['flush_rows']}",
            "C": mc["C"], "g": mc["g"], "mode": mc["mode"],
            "flush_rows": mc["flush_rows"], "timestamps": mc.get("timestamps"),
            "soak_iters": rb["soak_iters"], "n_rank_files": rb["n_rank_files"],
            "cfg_consistent_across_all_ranks": rb["consistent"],
            "status": c["status"], "run_count": c["run_count"],
            "mps_p50": c["arm_p50_median"]["mps_mega"],
            "prod_p50": c["arm_p50_median"]["production"],
            "pf6gm_p50": c["arm_p50_median"].get("pf6gm_mega"),
            "ratio": c["arm_p50_median"]["mps_mega"] / c["arm_p50_median"]["production"],
            "rotation_mps_p50": runs_mps, "rotation_prod_p50": runs_prod,
            "gates": {k: g.get(k) for k in (
                "mok_gate", "mok_gate_pass", "control_fails", "pperr_max",
                "soak_pass", "soak_epochs", "poison_selftest_all_true",
                "poison_nonfinite", "poison_survivors_max", "poison_lines_n",
                "spin_success_max", "spin_fail_max", "spin_lines_n")},
            "stamps": {
                "plan_M3toM5": (g.get("ts_split_us") or {}).get("plan_M3toM5"),
                "M6": (g.get("ts_split_us") or {}).get("M6"),
                "planM6": (g.get("ts_delta_us") or {}).get("planM6"),
                "M7": (g.get("ts_delta_us") or {}).get("M7"),
                "combine": (g.get("ts_delta_us") or {}).get("combine"),
                "servicedrain": (g.get("ts_delta_us") or {}).get("servicedrain"),
                "m2_to_end": (g.get("ts_delta_us") or {}).get("m2_to_end"),
            },
            "stamps_all_rotations": {
                "M6": g.get("ts_split_all_M6_us") or [],
                "M7": g.get("ts_delta_all_M7_us") or [],
                "combine": g.get("ts_delta_all_combine_us") or [],
                "planM6": g.get("ts_delta_all_planM6_us") or [],
            },
        })

    camps.sort(key=lambda c: (c["tag"], c["idx"]))
    for c in camps:
        g = c["gates"]
        c["gates_green"] = bool(
            c["status"] == "valid_diagnostic" and g["mok_gate_pass"] == "True"
            and g["control_fails"] == "True" and g["pperr_max"] == 0
            and g["soak_pass"] == "True" and g["soak_epochs"] == "600/600"
            and g["poison_selftest_all_true"] is True
            and g["poison_survivors_max"] == 0
            and g["poison_nonfinite"] == [57344]
            and c["soak_iters"] == [600] and c["run_count"] == 5
            and c["cfg_consistent_across_all_ranks"])
        # batch e37a is 4 interleave rounds of 3 permuted arms; e37b is 2 of 3
        c["round"] = ((c["idx"] - 1) // 3) + 1

    arms = {}
    for c in camps:
        arms.setdefault(c["arm"], []).append(c)

    # ---------------- paired comparisons, within a batch, by round -------------
    def paired(a_name, b_name):
        A = {(c["tag"], c["round"]): c for c in arms.get(a_name, [])}
        B = {(c["tag"], c["round"]): c for c in arms.get(b_name, [])}
        keys = sorted(set(A) & set(B))
        if not keys:
            return None
        dus = [A[k]["mps_p50"] - B[k]["mps_p50"] for k in keys]
        drt = [A[k]["ratio"] - B[k]["ratio"] for k in keys]
        xa = [v for c in arms[a_name] for v in c["rotation_mps_p50"]]
        xb = [v for c in arms[b_name] for v in c["rotation_mps_p50"]]
        rot_t = None
        if len(xa) > 1 and len(xb) > 1:
            se = math.sqrt(statistics.variance(xa) / len(xa)
                           + statistics.variance(xb) / len(xb))
            rot_t = (mean(xa) - mean(xb)) / se if se else None
        u = ci95(dus)
        r = ci95(drt)
        return {
            "a": a_name, "b": b_name,
            "pairing": "same batch, same interleave round (campaign = cluster)",
            "pairs": [f"{k[0]}#r{k[1]}" for k in keys],
            "per_pair_delta_us": [round(x, 1) for x in dus],
            "n_pairs": u["n"],
            "delta_us_mean": round(u["mean"], 2),
            "delta_us_sd": None if u["sd"] is None else round(u["sd"], 2),
            "delta_us_ci95": None if u["ci95"] is None else [round(x, 2) for x in u["ci95"]],
            "t_clustered": None if u["t"] is None else round(u["t"], 2),
            "df": u["df"],
            "sign_test_p_two_sided": sign_test_p(dus),
            "ranksum_exact_p_two_sided": ranksum_exact_p(
                [c["mps_p50"] for c in arms[a_name]],
                [c["mps_p50"] for c in arms[b_name]]),
            "campaign_values_disjoint": (max(c["mps_p50"] for c in arms[a_name])
                                         < min(c["mps_p50"] for c in arms[b_name]))
                                        or (min(c["mps_p50"] for c in arms[a_name])
                                            > max(c["mps_p50"] for c in arms[b_name])),
            "delta_ratio_mean": round(r["mean"], 5),
            "delta_ratio_ci95": None if r["ci95"] is None else [round(x, 5) for x in r["ci95"]],
            "INVALID_unclustered_rotation_t": None if rot_t is None else round(rot_t, 2),
            "INVALID_unclustered_note": (
                "rotation-level Welch t on 5x more 'samples'; reported only to "
                "show the inflation the clustered test avoids"),
        }

    names = sorted(arms, key=lambda a: (arms[a][0]["mode"], arms[a][0]["C"]))
    comparisons = {}
    for a, b in combinations(names, 2):
        p = paired(a, b)
        if p:
            comparisons[f"{a}__vs__{b}"] = p

    # ------------------------------- arm records ------------------------------
    def stamp_block(cs, key):
        per_camp = [c["stamps"][key] for c in cs if c["stamps"].get(key) is not None]
        rot = [v for c in cs for v in c["stamps_all_rotations"].get(key, [])]
        blk = {"per_campaign_final_epoch": per_camp,
               "median_of_campaigns": round(statistics.median(per_camp), 1) if per_camp else None}
        if rot:
            blk.update({"n_rotation_samples": len(rot),
                        "rotation_mean": round(mean(rot), 1),
                        "rotation_median": round(statistics.median(rot), 1),
                        "rotation_sd": round(statistics.stdev(rot), 1) if len(rot) > 1 else None,
                        "rotation_min": round(min(rot), 1),
                        "rotation_max": round(max(rot), 1)})
        return blk

    arm_records = []
    for a in names:
        cs = arms[a]
        c0 = cs[0]
        mps = [c["mps_p50"] for c in cs]
        prod = [c["prod_p50"] for c in cs]
        rat = [c["ratio"] for c in cs]
        key = f"{a}__vs__{CTRL}"
        rev = f"{CTRL}__vs__{a}"
        pc = comparisons.get(key)
        if pc is None and rev in comparisons:
            src = comparisons[rev]
            pc = dict(src)
            pc["a"], pc["b"] = src["b"], src["a"]
            pc["per_pair_delta_us"] = [-x for x in src["per_pair_delta_us"]]
            pc["delta_us_mean"] = -src["delta_us_mean"]
            pc["delta_us_ci95"] = (None if src["delta_us_ci95"] is None
                                   else sorted(-x for x in src["delta_us_ci95"]))
            pc["t_clustered"] = (None if src["t_clustered"] is None else -src["t_clustered"])
            pc["delta_ratio_mean"] = -src["delta_ratio_mean"]
            pc["delta_ratio_ci95"] = (None if src["delta_ratio_ci95"] is None
                                      else sorted(-x for x in src["delta_ratio_ci95"]))
            pc["INVALID_unclustered_rotation_t"] = (
                None if src["INVALID_unclustered_rotation_t"] is None
                else -src["INVALID_unclustered_rotation_t"])
        arm_records.append({
            "arm_id": a, "mode": c0["mode"], "C": c0["C"], "g": c0["g"],
            "flush_rows": c0["flush_rows"], "status": "measured",
            "n_campaigns": len(cs),
            "campaigns": [round(x, 1) for x in mps],
            "campaign_ids": [f"{c['tag']}#{c['idx']}" for c in cs],
            "interleave_rounds": [f"{c['tag']}#r{c['round']}" for c in cs],
            "p50_median": round(statistics.median(mps), 1),
            "p50_mean": round(mean(mps), 1),
            "p50_min": round(min(mps), 1), "p50_max": round(max(mps), 1),
            "rotation_p50_us": [[round(v, 1) for v in c["rotation_mps_p50"]] for c in cs],
            "production_p50_same_run": [round(x, 1) for x in prod],
            "production_p50_same_run_median": round(statistics.median(prod), 1),
            "ratio_vs_production": round(statistics.median(rat), 4),
            "ratio_vs_production_per_campaign": [round(x, 4) for x in rat],
            "delta_vs_ratchet_us": (0.0 if a == CTRL
                                    else (None if pc is None else pc["delta_us_mean"])),
            "delta_vs_ratchet_note": (
                "paired against the C=16 ratchet campaign from the SAME interleave "
                "round; the standing published ratchet number is "
                f"{RATCHET_US} us from a different session"),
            "paired_ci": pc if a != CTRL else "control arm (paired denominator)",
            "phase_stamps": {
                "plan": stamp_block(cs, "plan_M3toM5"),
                "M6": stamp_block(cs, "M6"),
                "M7": stamp_block(cs, "M7"),
                "combine": stamp_block(cs, "combine"),
                "servicedrain": stamp_block(cs, "servicedrain"),
                "units": "us",
                "note": ("device stamps, 1 tick = 0.01 us, emitted once per "
                         "rotation from the FINAL SOAK epoch. Phase SHAPE only: "
                         "they do not sum to arm_p50_us."),
            },
            "spin_success_max": max(c["gates"]["spin_success_max"] for c in cs),
            "spin_fail_max": max(c["gates"]["spin_fail_max"] for c in cs),
            "gates_green": all(c["gates_green"] for c in cs),
            "n_campaigns_gates_green": sum(1 for c in cs if c["gates_green"]),
        })

    # ------------------------- arms that could not be run ---------------------
    arm_records.append({
        "arm_id": "mode12_C0_g353_fr16", "mode": 12, "C": 0, "g": 353,
        "flush_rows": 16, "status": "requires_mode_14", "n_campaigns": 0,
        "campaigns": [], "p50_median": None, "production_p50_same_run": [],
        "ratio_vs_production": None, "delta_vs_ratchet_us": None,
        "paired_ci": None, "phase_stamps": None, "spin_success_max": None,
        "spin_fail_max": None, "gates_green": None,
        "note": ("C=0 is ILLEGAL in mode 12 -- the adapter validator rejects it "
                 "(moe_mps_adapter.cuh:373/375). Not faked with a large-C proxy. "
                 "The zero-dedication point needs mode 14."),
    })
    for c in (0, 8, 16):
        arm_records.append({
            "arm_id": f"mode14_C{c}", "mode": 14, "C": c, "g": None,
            "flush_rows": 16, "status": "pending_exp_34", "n_campaigns": 0,
            "campaigns": [], "p50_median": None, "production_p50_same_run": [],
            "ratio_vs_production": None, "delta_vs_ratchet_us": None,
            "paired_ci": None, "phase_stamps": None, "spin_success_max": None,
            "spin_fail_max": None, "gates_green": None,
            "note": ("mode 14 had not landed on the branch at this pin; not run "
                     "on the profiler's own initiative."),
        })

    # ------------------- phase attribution, clustered the same way ------------
    def camp_stamp_mean(c, k):
        v = c["stamps_all_rotations"].get(k) or []
        return mean(v) if v else None

    def paired_stamp(a_name, b_name, key):
        A = {(c["tag"], c["round"]): c for c in arms.get(a_name, [])}
        B = {(c["tag"], c["round"]): c for c in arms.get(b_name, [])}
        ks = sorted(set(A) & set(B))
        ds = []
        for k in ks:
            xa, xb = camp_stamp_mean(A[k], key), camp_stamp_mean(B[k], key)
            if xa is None or xb is None:
                continue
            ds.append(xa - xb)
        if len(ds) < 2:
            return None
        s = ci95(ds)
        return {"n_pairs": s["n"], "delta_us_mean": round(s["mean"], 1),
                "delta_us_ci95": [round(x, 1) for x in s["ci95"]],
                "spans_zero": s["ci95"][0] * s["ci95"][1] <= 0,
                "per_pair": [round(x, 1) for x in ds]}

    def paired_stamp_sum(a_name, b_name, keys):
        A = {(c["tag"], c["round"]): c for c in arms.get(a_name, [])}
        B = {(c["tag"], c["round"]): c for c in arms.get(b_name, [])}
        ks = sorted(set(A) & set(B))
        ds = []
        for k in ks:
            xa = [camp_stamp_mean(A[k], x) for x in keys]
            xb = [camp_stamp_mean(B[k], x) for x in keys]
            if any(v is None for v in xa + xb):
                continue
            ds.append(sum(xa) - sum(xb))
        if len(ds) < 2:
            return None
        s = ci95(ds)
        return {"n_pairs": s["n"], "delta_us_mean": round(s["mean"], 1),
                "delta_us_ci95": [round(x, 1) for x in s["ci95"]],
                "spans_zero": s["ci95"][0] * s["ci95"][1] <= 0,
                "per_pair": [round(x, 1) for x in ds]}

    attribution = {
        "unit": "campaign mean over its 5 rotation stamps; paired by interleave round",
        "question": "which phase pays for the C change?",
        "comparisons": {},
    }
    for cand in ("mode12_C4_g353_fr16", "mode12_C8_g353_fr16"):
        if cand not in arms:
            continue
        attribution["comparisons"][f"{cand}__vs__{CTRL}"] = {
            "M6": paired_stamp(cand, CTRL, "M6"),
            "M7": paired_stamp(cand, CTRL, "M7"),
            "combine": paired_stamp(cand, CTRL, "combine"),
            "M7_plus_combine": paired_stamp_sum(cand, CTRL, ("M7", "combine")),
            "planM6": paired_stamp(cand, CTRL, "planM6"),
        }
    if "mode12_C4_g353_fr16" in arms and "mode12_C8_g353_fr16" in arms:
        attribution["comparisons"]["mode12_C4_g353_fr16__vs__mode12_C8_g353_fr16"] = {
            "M6": paired_stamp("mode12_C4_g353_fr16", "mode12_C8_g353_fr16", "M6"),
            "M7": paired_stamp("mode12_C4_g353_fr16", "mode12_C8_g353_fr16", "M7"),
            "combine": paired_stamp("mode12_C4_g353_fr16", "mode12_C8_g353_fr16", "combine"),
            "M7_plus_combine": paired_stamp_sum(
                "mode12_C4_g353_fr16", "mode12_C8_g353_fr16", ("M7", "combine")),
        }

    prods = [c["prod_p50"] for c in camps]
    prod_spread = 100 * (max(prods) - min(prods)) / min(prods)

    payload = {
        "schema_version": "exp_37.placement.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "experiment": "exp_37 placement adjudication (paper Fig 5 / Q2)",
        "host": raw.get("host"),
        "harness": {
            "campaign": "5 rotations, 500 warmup / 100 timed, K0_MPS_SOAK_ITERS=600",
            "arms_per_run": "production, pf6gm_mega, mps_mega (same-run denominator)",
            "T": 4096,
            "stat_unit": "campaign (median of its 5 rotation p50s)",
            "clustering": ("inference clusters by CAMPAIGN; rank-level and "
                           "rotation-level samples are NOT treated as independent"),
        },
        "caveats": [
            "All arms carry timestamps=1, so the stamp instrumentation is a "
            "constant across every arm; absolute microseconds are therefore not "
            "directly comparable to no-stamp numbers from other experiments, but "
            "every comparison inside this file is like-for-like.",
            "Phase stamps are sampled from the final SOAK epoch of each rotation, "
            "not from a timed iteration; use them for phase SHAPE, not for a "
            "budget that sums to the end-to-end number.",
            "C=0 in mode 12 is rejected by the validator; the true zero-dedication "
            "point is only reachable in mode 14, which had not landed at this pin.",
            "Session drift is bounded by the same-run production denominator, "
            f"which moved only {prod_spread:.2f}% peak-to-peak across the whole "
            "session (27 campaigns, ~1.5 h).",
            "The M7/combine BOUNDARY is not stable across C: M7 alone and "
            "combine alone move in opposite directions and each spans zero for "
            "at least one candidate, while their SUM is significant for both. "
            "Attribute the C effect to the payload-carrying pair (M7+combine), "
            "not to M7 alone.",
            "The mori jit cache holds two byte-different builds of "
            "k0pf6gm_mps_mega.hsaco and the latest/ symlink flips between them "
            "run to run, uncorrelated with the arm. They differ in 40 of 188360 "
            "bytes, all of it the HIP compilation-unit id (__hip_cuid_...), so "
            "the flip is cosmetic and is not a confound. summary.json's "
            "kernel_hsaco_sha256 is an empty dict on this harness and cannot be "
            "used as build evidence.",
            "C=12, C=32 and the mode-2 dedicated pool have n=2 campaigns each: "
            "they are figure points for the shape of the axis, and their "
            "replicates are tight (<=2.4 us apart), but they were not run to "
            "ratchet strength.",
        ],
        "phase_attribution": attribution,
        "arms": arm_records,
        "pairwise_comparisons": comparisons,
        "campaigns_raw": camps,
    }
    (outdir / "placement.json").write_text(json.dumps(payload, indent=1))

    # ------------------------------------------------------------------ report
    print(f"{'arm':<24}{'n':>3}{'p50 med':>10}{'mean':>9}{'min':>9}{'max':>9}"
          f"{'ratio':>8}{'M6':>9}{'M7':>9}{'cmb':>8}  green")
    for r in arm_records:
        if r["status"] != "measured":
            print(f"{r['arm_id']:<24}  -  {r['status']}")
            continue
        ps = r["phase_stamps"]
        print(f"{r['arm_id']:<24}{r['n_campaigns']:>3}{r['p50_median']:>10}"
              f"{r['p50_mean']:>9}{r['p50_min']:>9}{r['p50_max']:>9}"
              f"{r['ratio_vs_production']:>8}"
              f"{ps['M6']['rotation_mean']:>9}{ps['M7']['rotation_mean']:>9}"
              f"{ps['combine']['rotation_mean']:>8}  {r['gates_green']}")
    print()
    for k, p in comparisons.items():
        print(f"{k}\n    n={p['n_pairs']} delta={p['delta_us_mean']:+.1f} us "
              f"ci95={p['delta_us_ci95']} sd={p['delta_us_sd']} "
              f"t={p['t_clustered']} sign_p={p['sign_test_p_two_sided']} "
              f"ranksum_p={p['ranksum_exact_p_two_sided']} "
              f"disjoint={p['campaign_values_disjoint']} "
              f"per_pair={p['per_pair_delta_us']} "
              f"[INVALID unclustered t={p['INVALID_unclustered_rotation_t']}]")
    print()
    bad = [c for c in camps if not c["gates_green"]]
    print(f"campaigns: {len(camps)}  gates green: {len(camps) - len(bad)}")
    for c in bad:
        print("  NOT GREEN:", c["tag"], c["idx"], c["arm"], c["status"], c["gates"])
    prods = [c["prod_p50"] for c in camps]
    print(f"production denominator across session: min={min(prods):.1f} "
          f"max={max(prods):.1f} spread={100*(max(prods)-min(prods))/min(prods):.2f}%")
    print(f"wrote {outdir/'placement.json'}")


if __name__ == "__main__":
    main()
