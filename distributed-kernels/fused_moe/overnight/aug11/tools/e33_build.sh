#!/usr/bin/env bash
# exp_33 builder: turn the campaign logs + summary.json + rank JSONs into
# phase_stamps.json (plot-ready). Also tars the raw evidence bundle.
# Usage: e33_build.sh TAG1 [TAG2 ...]
set -uo pipefail
cd "$HOME" || exit 1
mkdir -p "$HOME/e33/out"

python3 - "$@" <<'PY'
import glob, json, math, os, re, statistics as st, sys, time

HOME = os.path.expanduser("~")
TICK_US = 0.01
tags = sys.argv[1:]

def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0

def agg(vals):
    """mean / sample-sd / stderr / n for a list of per-rotation samples."""
    if not vals:
        return None
    n = len(vals)
    s = sd(vals)
    return {"mean": round(st.mean(vals), 2),
            "sd": round(s, 2),
            "stderr": round(s / math.sqrt(n), 2) if n > 1 else None,
            "n": n,
            "values": [round(x, 2) for x in vals]}

# ---------------------------------------------------------------- campaigns
campaigns = []
for tag in tags:
    outs = sorted(glob.glob(f"{HOME}/k0-mok-{tag}/*/summary.json"))
    if not outs:
        print(f"WARN no summary.json for tag {tag}", file=sys.stderr)
        continue
    sjp = outs[-1]
    outdir = os.path.dirname(sjp)
    logs = sorted(glob.glob(f"{HOME}/overnight-scratch/{tag}_*.log"))
    campaigns.append({"tag": tag, "outdir": outdir, "summary": sjp,
                      "log": logs[-1] if logs else None})

TS_RE = re.compile(r"\[MPS TS\] (.+)")
SPLIT_RE = re.compile(r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)")
DELTA_RE = re.compile(r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
                      r"servicedrain=(-?\d+) m2_to_end=(-?\d+)")
SPIN_RE = re.compile(r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+) limit=(\d+)")
RUNSTART_RE = re.compile(r"starting run=(\d+)/(\d+) arms=(\S+)")
GATE_RE = re.compile(r"\[MOK GATE\] (\S+) max_abs=(\S+) relative=(\S+) pass=(\w+)")
SOAK_RE = re.compile(r"\[MPS SOAK\] completed=(\d+)/(\d+) pperr=(\d+) poison=(\d+) "
                     r"poison_epoch=(-?\d+) pass=(\w+)")
SELF_RE = re.compile(r"\[POISON SELFTEST\] arm=(\S+) one_row_poisoned_fails=(\w+) nonfinite=(\d+)")
POIS_RE = re.compile(r"\[POISON\] (\S+) arm=(\S+) survivors=(\d+)")
PPERR_RE = re.compile(r"pperr=(\d+)")

# ------------------------------------------------------- parse the logs
rot = []          # one record per rotation, in run order, across all campaigns
gates = {"mok_gate_fail": [], "control_fails": set(), "soak": [],
         "selftest": set(), "poison_survivors": set(), "pperr_max": 0,
         "mok_gate_pass_counts": {}}
for c in campaigns:
    if not c["log"]:
        continue
    txt = open(c["log"], errors="ignore").read()
    run_idx = 0
    cur = None
    for ln in txt.splitlines():
        m = RUNSTART_RE.search(ln)
        if m:
            run_idx = int(m.group(1))
            cur = {"campaign": c["tag"], "run": run_idx, "arm_order": m.group(3)}
            continue
        m = TS_RE.search(ln)
        if m and cur is not None:
            kv = dict(p.split("=", 1) for p in m.group(1).split() if "=" in p)
            cur["raw_stamps"] = {k: int(v) for k, v in kv.items()}
            continue
        m = SPLIT_RE.search(ln)
        if m and cur is not None:
            cur["plan_M3toM5_us"] = int(m.group(1)) * TICK_US
            cur["M6_us"] = int(m.group(2)) * TICK_US
            continue
        m = DELTA_RE.search(ln)
        if m and cur is not None:
            cur["planM6_us"] = int(m.group(1)) * TICK_US
            cur["M7_us"] = int(m.group(2)) * TICK_US
            cur["combine_us"] = int(m.group(3)) * TICK_US
            cur["servicedrain_us"] = int(m.group(4)) * TICK_US
            cur["m2_to_end_us"] = int(m.group(5)) * TICK_US
            continue
        m = SPIN_RE.search(ln)
        if m and cur is not None:
            cur["spin_success_max"] = int(m.group(1))
            cur["spin_fail_max"] = int(m.group(2))
            cur["spin_limit"] = int(m.group(3))
            rot.append(cur)          # SPIN is the last stamp line of a rotation
            continue
        m = GATE_RE.search(ln)
        if m:
            arm, p = m.group(1), m.group(4)
            gates["mok_gate_pass_counts"].setdefault(arm, {"True": 0, "other": 0})
            gates["mok_gate_pass_counts"][arm]["True" if p == "True" else "other"] += 1
            if p != "True":
                gates["mok_gate_fail"].append(ln.strip())
            continue
        if "[MARK] control_fails=" in ln:
            gates["control_fails"].add(ln.strip().split("=")[-1])
        m = SOAK_RE.search(ln)
        if m:
            gates["soak"].append({"completed": f"{m.group(1)}/{m.group(2)}",
                                  "pperr": int(m.group(3)), "poison": int(m.group(4)),
                                  "poison_epoch": int(m.group(5)), "pass": m.group(6)})
        m = SELF_RE.search(ln)
        if m:
            gates["selftest"].add((m.group(1), m.group(2), m.group(3)))
        m = POIS_RE.search(ln)
        if m:
            gates["poison_survivors"].add(int(m.group(3)))
        for v in PPERR_RE.findall(ln):
            gates["pperr_max"] = max(gates["pperr_max"], int(v))

# ------------------------------------------------- summary.json + rank JSONs
arm_names = ["production", "pf6gm_mega", "mps_mega"]
p50_vals = {a: [] for a in arm_names}
p50_med = {}
prod_stage = {"maxrank": {}, "local": {}}
pf_full_stage = {"maxrank": {}, "local": {}}
rank0_e2e = {a: [] for a in arm_names}
build_ids = set()
meta = {}
for c in campaigns:
    S = json.load(open(c["summary"]))
    for a in arm_names:
        try:
            p50_vals[a] += [float(x) for x in S["arm_p50_us"][a]["values"]]
            p50_med.setdefault(a, []).append(float(S["arm_p50_us"][a]["median"]))
        except Exception:
            pass
    c["kernel_hsaco_sha256"] = S.get("kernel_hsaco_sha256")
    c["kernel_source_sha256"] = S.get("kernel_source_sha256")
    c["run_count"] = S.get("run_count")
    c["arm_p50_median"] = {a: round(float(S["arm_p50_us"][a]["median"]), 2)
                           for a in arm_names if a in S.get("arm_p50_us", {})}
    meta.setdefault("run_count", S.get("run_count"))
    for rj in sorted(glob.glob(os.path.dirname(c["summary"]) + "/run*/k0pf_mok_synthetic_rank0.json")):
        D = json.load(open(rj))
        spr = D.get("stage_profile", {})
        for key, dest in (("production", prod_stage), ("pf_full", pf_full_stage)):
            blk = spr.get(key, {})
            for which, jkey in (("maxrank", "maxrank_us_p50"), ("local", "local_us_p50")):
                for st_name, v in (blk.get(jkey) or {}).items():
                    dest[which].setdefault(st_name, []).append(float(v))
        for a in arm_names:
            loc = ((D.get("mok_eager") or {}).get("arms") or {}).get(a, {}).get("local_us")
            if loc:
                rank0_e2e[a].append(float(st.median(loc)))
        pmps = D.get("pf6mps") or {}
        build_ids.add((c["tag"],
                       (pmps.get("hsaco_path") or {}).get("k0pf6gm_mps_mega", "").split("/")[-2],
                       (pmps.get("hsaco_sha256") or {}).get("k0pf6gm_mps_mega"),
                       (pmps.get("source_sha256") or {}).get("k0pf6gm_mps_mega.hip")))
        meta.setdefault("mps_config", D.get("mps_config"))
        meta.setdefault("warmup_iters", (D.get("mok_eager") or {}).get("warmup_iters"))
        meta.setdefault("timed_iters", (D.get("mok_eager") or {}).get("timed_iters"))
        meta.setdefault("soak_iters", (D.get("config") or {}).get("mps_soak_iters"))

# --------------------------------------------------------------- assemble
def series(key):
    return [r[key] for r in rot if key in r]

prod_p50 = st.median(p50_vals["production"]) if p50_vals["production"] else None
prod_p50_rank0 = st.median(rank0_e2e["production"]) if rank0_e2e["production"] else None

NO_RANKMAX = ("unavailable by construction: the [MPS TS]/[MPS SPIN] block in "
              "e004pf_k0pf_ab.py is inside `if rank == 0:` and reads "
              "pf6_state['mps_state'], which is never all-reduced and never "
              "written to the per-rank JSON. The value given is the device "
              "CTA-max WITHIN rank 0, not a cross-rank max.")
NO_PF6GM = ("unavailable by construction: pf6gm_mega is a different kernel "
            "(k0pf6gm_mega, descriptor desc_gm) and is not passed the mps_state "
            "timestamp block, so it emits no phase stamps. The only per-phase "
            "instrument for pf6gm_mega is the K0_PF6GM_DECOMP cumulative-prefix "
            "path (ab.py:5228), and K0_PF6GM_DECOMP is absent from "
            "run_campaign.sh's -e forwarding list, so it cannot be reached "
            "through the campaign. Enabling it is a one-line harness edit that "
            "exp_33 does not own.")

_interior_mean = agg(series("m2_to_end_us"))["mean"] if series("m2_to_end_us") else None
_mps_p50 = st.median(p50_vals["mps_mega"]) if p50_vals["mps_mega"] else None

def pct(x, of):
    return round(100.0 * x / of, 2) if (x is not None and of) else None

def phase(name, key, source):
    a = agg(series(key))
    if a is None:
        return None
    return {"phase": name, "source": source,
            "us_rank_max": None, "us_rank_max_reason": NO_RANKMAX,
            "us_rank0": a["mean"], "n": a["n"], "stderr": a["stderr"],
            "sd": a["sd"], "values_us": a["values"],
            "share_of_interior_pct": pct(a["mean"], _interior_mean),
            "share_of_p50_pct": pct(a["mean"], _mps_p50),
            "stderr_pct_of_mean": pct(a["stderr"], a["mean"]) if a["stderr"] else None,
            "us_rank0_is_cta_max_within_rank0": True}

mps_phases = {
    "dispatch_M0toM2": {
        "phase": "dispatch_M0toM2", "source": "residual (not stamped)",
        "us_rank_max": None, "us_rank_max_reason": NO_RANKMAX,
        "us_rank0": None, "n": 0, "stderr": None,
        "reason": ("the 8-slot stamp layout has no kernel-start stamp: the "
                   "former KSTART slot now holds M5_DONE, and FIRST_READY_inv "
                   "is a MIN stamp over all 600 soak epochs so it cannot be "
                   "differenced against final-epoch stamps. See "
                   "derived.dispatch_M0toM2_residual_estimate_us for the "
                   "instrument-mixed residual."),
    },
    "plan_M3toM5": phase("plan_M3toM5", "plan_M3toM5_us", "[MPS TS SPLIT] plan_M3toM5 = M5_DONE - M2_DONE"),
    "M6_gemm1": phase("M6_gemm1", "M6_us", "[MPS TS SPLIT] M6 = M6_DONE - M5_DONE"),
    "M7_gemm2_total": phase("M7_gemm2_total", "M7_us", "[MPS TS DELTA] M7 = M7_DONE - M6_DONE"),
    "M7_gemm_proper": {
        "phase": "M7_gemm_proper", "source": "not measurable in this arm",
        "us_rank_max": None, "us_rank0": None, "n": 0, "stderr": None,
        "reason": ("requires M7 of pf6gm_mega as the payload-free reference; "
                   + NO_PF6GM),
    },
    "M7_epilogue_surcharge": {
        "phase": "M7_epilogue_surcharge", "source": "M7(mps_mega) - M7(pf6gm_mega)",
        "us_rank_max": None, "us_rank0": None, "n": 0, "stderr": None,
        "reason": ("the specified same-arm difference is not computable: "
                   + NO_PF6GM
                   + " See derived.m7_epilogue_surcharge_proxy_us for two "
                     "cross-path proxy estimates."),
    },
    "combine_M8M9": phase("combine_M8M9", "combine_us", "[MPS TS DELTA] combine = REDUCE_DONE - M7_DONE"),
    "service_drain": phase("service_drain", "servicedrain_us", "[MPS TS DELTA] servicedrain = DRAIN - M6_DONE (overlaps M7/combine; NOT additive)"),
    "interior_M3toM9": phase("interior_M3toM9", "m2_to_end_us", "[MPS TS DELTA] m2_to_end = REDUCE_DONE - M2_DONE (sum of plan+M6+M7+combine)"),
}

prod_phases = {}
_pmap = {"dispatch": "dispatch", "gemm": "gemm", "combine": "combine"}
for st_name in ("dispatch", "gemm", "combine"):
    mx = agg(prod_stage["maxrank"].get(st_name, []))
    lc = agg(prod_stage["local"].get(st_name, []))
    if mx is None:
        continue
    prod_phases[st_name] = {
        "phase": st_name,
        "source": "[K0PF PROFILE] production / rank JSON stage_profile.production (HIP-event, K0_STAGE_REPS=5 eager reps, per-stage dist MAX over ranks)",
        "us_rank_max": mx["mean"], "us_rank0": lc["mean"] if lc else None,
        "n": mx["n"], "stderr": mx["stderr"], "sd": mx["sd"],
        "stderr_rank0": lc["stderr"] if lc else None,
        "stderr_pct_of_mean": pct(mx["stderr"], mx["mean"]) if mx["stderr"] else None,
        "values_us_rank_max": mx["values"],
        "values_us_rank0": lc["values"] if lc else None,
    }
_prod_p50 = st.median(p50_vals["production"]) if p50_vals["production"] else None
_prod_p50_r0 = st.median(rank0_e2e["production"]) if rank0_e2e["production"] else None
_psum_mx = round(sum(v["us_rank_max"] for v in prod_phases.values()), 2) if prod_phases else None
_psum_r0 = round(sum(v["us_rank0"] for v in prod_phases.values()), 2) if prod_phases else None
prod_stage_sums = {
    "stage_sum_us_rank_max": _psum_mx,
    "stage_sum_us_rank0": _psum_r0,
    "stage_sum_vs_p50_pct_rank_max": pct(_psum_mx - _prod_p50, _prod_p50) if (_psum_mx and _prod_p50) else None,
    "stage_sum_vs_p50_pct_rank0": pct(_psum_r0 - _prod_p50_r0, _prod_p50_r0) if (_psum_r0 and _prod_p50_r0) else None,
    "note": ("the three rank-max stages sum ABOVE production's p50 because the rank "
             "MAX is taken independently per stage (max-of-sums <= sum-of-maxes); the "
             "rank-0 sum is much closer to the rank-0 p50"),
    "combine_rank_spread_pct": pct(prod_phases["combine"]["us_rank_max"]
                                   - prod_phases["combine"]["us_rank0"],
                                   prod_phases["combine"]["us_rank0"]) if "combine" in prod_phases else None,
}

pf6gm_phases = {k: {"phase": k, "us_rank_max": None, "us_rank0": None,
                    "n": 0, "stderr": None, "reason": NO_PF6GM}
                for k in ("dispatch_M0toM2", "plan_M3toM5", "M6_gemm1",
                          "M7_gemm2_total", "M7_gemm_proper",
                          "M7_epilogue_surcharge", "combine_M8M9")}

# ------------------------------------------------------------- derived
m7 = series("M7_us"); cb = series("combine_us")
coupled = [a + b for a, b in zip(m7, cb)]
def pearson(x, y):
    if len(x) < 3:
        return None
    mx_, my_ = st.mean(x), st.mean(y)
    num = sum((a - mx_) * (b - my_) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx_) ** 2 for a in x) * sum((b - my_) ** 2 for b in y))
    return round(num / den, 4) if den else None

mps_p50 = st.median(p50_vals["mps_mega"]) if p50_vals["mps_mega"] else None
interior = agg(series("m2_to_end_us"))
pf_n2p2_mx = agg(pf_full_stage["maxrank"].get("n2_p2", []))
pf_n2p2_lc = agg(pf_full_stage["local"].get("n2_p2", []))
m7_agg = agg(m7)

derived = {
    "identity_check_us": {
        "note": "plan_M3toM5 + M6 + M7 + combine must equal m2_to_end exactly",
        "sum_of_parts_mean": round(sum(agg(series(k))["mean"] for k in
                                      ("plan_M3toM5_us", "M6_us", "M7_us", "combine_us")), 2),
        "m2_to_end_mean": interior["mean"] if interior else None,
        "max_abs_per_rotation_residual_us": round(max(
            abs(r["plan_M3toM5_us"] + r["M6_us"] + r["M7_us"] + r["combine_us"]
                - r["m2_to_end_us"]) for r in rot), 4) if rot else None,
    },
    "dispatch_M0toM2_residual_estimate_us": {
        "value": round(mps_p50 - interior["mean"], 2) if (mps_p50 and interior) else None,
        "share_of_p50_pct": pct(mps_p50 - interior["mean"], mps_p50) if (mps_p50 and interior) else None,
        "reference_archived_pf6gm_M0_M1_M2_us": 768.0,
        "reference_source": ("[PF6GM DECOMP] M0_retire_zero + M1_qpush + M2_stream_verify "
                             "~= 15.9 + 552.7 + 205.4 from the donor tree exp_58 archive "
                             "(2026-08-06, different commit) -- an order-of-magnitude sanity "
                             "check on the residual, not a measurement of this arm"),
        "formula": "arm_p50_us[mps_mega].median (timed iteration, rank-max) - interior_M3toM9 (final soak epoch, rank-0 CTA-max)",
        "caveat": ("INSTRUMENT-MIXED and DERIVED, not measured: it differences a "
                   "timed-iteration rank-max total against a soak-epoch rank-0 "
                   "interior. It absorbs M0+M1+M2, kernel launch, and epoch skew. "
                   "Treat as an upper bound on dispatch, good to ~100 us at best."),
    },
    "M7_plus_combine_coupled_block_us": {
        "value": agg(coupled)["mean"] if coupled else None,
        "stderr": agg(coupled)["stderr"] if coupled else None,
        "sd": agg(coupled)["sd"] if coupled else None,
        "n": len(coupled),
        "values": agg(coupled)["values"] if coupled else None,
        "share_of_interior_pct": pct(agg(coupled)["mean"], _interior_mean) if coupled else None,
        "share_of_p50_pct": pct(agg(coupled)["mean"], _mps_p50) if coupled else None,
        "stderr_pct_of_mean": pct(agg(coupled)["stderr"], agg(coupled)["mean"]) if coupled else None,
        "sd_if_independent_us": round(math.sqrt(agg(m7)["sd"] ** 2 + agg(cb)["sd"] ** 2), 2)
                                if (m7 and cb) else None,
        "sd_if_independent_note": ("sqrt(sd_M7^2 + sd_combine^2): what the sum's sd would be "
                                   "if the two phases were independent. The observed sd is far "
                                   "smaller, which is the quantitative form of the coupling."),
        "pearson_r_M7_vs_combine": pearson(m7, cb),
        "note": ("M7 and combine are strongly anti-correlated: their sum is far "
                 "tighter than either term, so the M7_DONE/REDUCE_DONE split "
                 "point is jitter, and M7+combine is the schedulable unit."),
    },
    "m7_epilogue_surcharge_proxy_us": {
        "specified_formula": "M7(mps_mega) - M7(pf6gm_mega)  [NOT COMPUTABLE, see arms.pf6gm_mega]",
        "proxy_A_same_run_pf_full_n2_p2": {
            "value": round(m7_agg["mean"] - pf_n2p2_mx["mean"], 2) if (m7_agg and pf_n2p2_mx) else None,
            "minuend_us": m7_agg["mean"] if m7_agg else None,
            "subtrahend_us": pf_n2p2_mx["mean"] if pf_n2p2_mx else None,
            "subtrahend_us_rank0": pf_n2p2_lc["mean"] if pf_n2p2_lc else None,
            "value_using_rank0_subtrahend": round(m7_agg["mean"] - pf_n2p2_lc["mean"], 2)
                                            if (m7_agg and pf_n2p2_lc) else None,
            "subtrahend_source": "stage_profile.pf_full.n2_p2 measured in THESE SAME runs (payload-free GEMM-2, the same donor n2_phase2 body pf6gm_mega's M7 runs)",
            "confidence": "medium",
        },
        "proxy_B_archived_pf6gm_decomp": {
            "value": None,
            "subtrahend_us": 1803.98,
            "subtrahend_source": ("[PF6GM DECOMP] M7_n2_phase2 archived in the donor tree "
                                  "(k0_fused_moe/experiments/exp_58_residual_attribution, 2026-08-06, "
                                  "a DIFFERENT commit) -- cross-run, not same-run"),
            "confidence": "low",
        },
        "subtrahend_agreement_pct": round(100.0 * abs(pf_n2p2_mx["mean"] - 1803.98)
                                          / 1803.98, 2) if pf_n2p2_mx else None,
        "surcharge_band_us": [round(m7_agg["mean"] - pf_n2p2_mx["mean"], 2),
                              round(m7_agg["mean"] - 1803.98, 2)] if (m7_agg and pf_n2p2_mx) else None,
        "surcharge_pct_of_M7_band": [pct(m7_agg["mean"] - pf_n2p2_mx["mean"], m7_agg["mean"]),
                                     pct(m7_agg["mean"] - 1803.98, m7_agg["mean"])]
                                    if (m7_agg and pf_n2p2_mx) else None,
        "caveat": ("Both proxies mix instruments: the minuend is a device wall-clock "
                   "delta from the final soak epoch, the subtrahend is a HIP-event "
                   "measure of a separately launched kernel with barriers outside "
                   "the window. Read the surcharge as 'several hundred us, order "
                   "700-900', not as a calibrated number."),
    },
    "gemm_dominance": {
        "M6_plus_M7_share_of_interior_pct": round(
            pct(agg(series("M6_us"))["mean"] + agg(series("M7_us"))["mean"], _interior_mean), 2)
            if series("M6_us") else None,
        "M6_plus_M7_us": round(agg(series("M6_us"))["mean"] + agg(series("M7_us"))["mean"], 2)
            if series("M6_us") else None,
        "note": "the two GEMM phases as a single share of the measured interior M3-M9",
    },
    "reproducibility": {
        "ratchet_recorded_us": 6482.7,
        "ratchet_recorded_source": "aug11 STATUS.md / commit ca5b683f subject (exp_27 ascale token-major)",
        "per_campaign_mps_p50_us": {c["tag"]: (c.get("arm_p50_median") or {}).get("mps_mega")
                                    for c in campaigns},
        "per_campaign_delta_vs_recorded_pct": {
            c["tag"]: pct((c.get("arm_p50_median") or {}).get("mps_mega", 0) - 6482.7, 6482.7)
            for c in campaigns},
        "between_campaign_spread_pct": {
            a: pct(max(p50_med.get(a, [0])) - min(p50_med.get(a, [0])), min(p50_med.get(a, [1])))
            for a in arm_names},
    },
    "first_ready_decoded": {
        "note": ("FIRST_READY_inv is stored complemented for min-tracking; "
                 "FIRST_READY = 2**64 - FIRST_READY_inv. It is a MIN over all "
                 "600 soak epochs, so it must NOT be differenced against the "
                 "final-epoch max stamps. Recorded raw only."),
        "per_rotation": [{"campaign": r["campaign"], "run": r["run"],
                          "first_ready_inv": r["raw_stamps"].get("FIRST_READY_inv"),
                          "first_ready_decoded": (2 ** 64) - r["raw_stamps"]["FIRST_READY_inv"]
                          if r.get("raw_stamps", {}).get("FIRST_READY_inv") else None}
                         for r in rot],
    },
}
if derived["m7_epilogue_surcharge_proxy_us"]["proxy_B_archived_pf6gm_decomp"]["subtrahend_us"] and m7_agg:
    derived["m7_epilogue_surcharge_proxy_us"]["proxy_B_archived_pf6gm_decomp"]["value"] = \
        round(m7_agg["mean"] - 1803.98, 2)

# ------------------------------------------------------------------ output
def arm_block(name, phases, extra=None):
    p50 = st.median(p50_vals[name]) if p50_vals[name] else None
    r0 = st.median(rank0_e2e[name]) if rank0_e2e[name] else None
    b = {
        "p50_us": round(p50, 2) if p50 else None,
        "p50_us_source": "summary.json arm_p50_us[arm]: median of index-aligned MAX(rank) p50 over rotations",
        "p50_us_values_per_rotation": [round(x, 2) for x in p50_vals[name]],
        "p50_us_per_campaign_median": [round(x, 2) for x in p50_med.get(name, [])],
        "p50_us_rank0": round(r0, 2) if r0 else None,
        "p50_us_rank0_source": "median over rotations of median(mok_eager.arms[arm].local_us) from rank0 JSON",
        "ratio_vs_production": round(p50 / prod_p50, 4) if (p50 and prod_p50) else None,
        "ratio_vs_production_rank0": round(r0 / prod_p50_rank0, 4) if (r0 and prod_p50_rank0) else None,
        "phases": phases,
    }
    if extra:
        b.update(extra)
    return b

doc = {
    "schema_version": "exp33-phase-stamps-1",
    "experiment": "exp_33 bottleneck attribution at the ratchet (paper Q3)",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "commit": {
        "repo_head": "ca5b683f420e4985e8d2a790fe086419ab6730f1",
        "branch": "codex/distributed-hipkittens-scaffold",
        "K0P6_MPS_SRC_REV": 26,
        "K0P6_MPS_ASCALE_TM": 1,
        "mps_build_identity_per_rotation": sorted(
            [{"campaign": t, "jit_dir": j, "hsaco_sha256": h, "source_sha256": s}
             for (t, j, h, s) in build_ids], key=lambda x: (x["campaign"], x["jit_dir"])),
        "build_identity_note": ("Authoritative identity is the in-container record "
                               "pf6mps.hsaco_sha256 in each rank JSON, NOT the host-side "
                               "`latest/` symlink stamp (which screen.sh samples outside "
                               "the run and which flipped between samples here). All 10 "
                               "rotations across both campaigns loaded ONE build "
                               "(jit 5635a2f2a370, hsaco a6e4189ce224..., source "
                               "e957bd00b48d...), so the two campaigns are true "
                               "replicates and pooling them is sound. summary.json's "
                               "kernel_hsaco_sha256/kernel_source_sha256 are empty dicts "
                               "in this harness build and must not be used."),
    },
    "config": {
        "K0_MPS_CFG": "C=16,g=353,mode=12,flush_rows=16,timestamps=1",
        "arms": arm_names,
        "K0_MOK_ARMS": "production,pf6gm_mega,mps_mega",
        "campaigns": [{"tag": c["tag"], "output_root": c["outdir"],
                       "log": c["log"], "run_count": c.get("run_count"),
                       "arm_p50_us_median": c.get("arm_p50_median")}
                      for c in campaigns],
        "rotations_total": len(rot),
        "warmup_iters": meta.get("warmup_iters"),
        "timed_iters": meta.get("timed_iters"),
        "soak_iters": meta.get("soak_iters"),
        "K0_MOK_POISON_OUT": 1,
        "K0_SPIN_LIMIT": rot[0].get("spin_limit") if rot else None,
        "mps_config_decoded": meta.get("mps_config"),
        "shape": {"K0_T": 4096, "K0_BLK": 128, "K0_WARP": 16, "K0_PF6GM_G": 3,
                  "route": "default (K0_SYNTH_ROUTE unset)", "seed": 0},
        "node": {"host": "gbt350-odcdh2-c05-1", "gpus": 8, "arch": "gfx950 (MI350X)"},
    },
    "units": {"tick_us": TICK_US,
              "tick_note": "1 device tick = 0.01 us (100 MHz wall clock; the 2.2 GHz shader clock is the WRONG divisor)"},
    "caveats": [
        "FINAL-SOAK-EPOCH: every [MPS TS] value is a running device MAX that is never reset, so after the 600-epoch soak the stamps describe the FINAL SOAK EPOCH, not a timed iteration. Do not sum them and compare to arm_p50_us; the phase table and the end-to-end p50 are different measurements of the same kernel.",
        "1 tick = 0.01 us. Every *_us number derived from stamps is ticks x 0.01.",
        "RANK-0 ONLY for every mps_mega phase: the stamp print block is inside `if rank == 0:` and the state is never all-reduced, so no rank-max variant exists. The reported value is the CTA-max within rank 0.",
        "pf6gm_mega emits NO phase stamps (different kernel, no mps_state), so the specified M7 epilogue surcharge M7(mps)-M7(pf6gm) is not computable in this campaign; two clearly-labelled cross-path proxies are given under derived.",
        "production's phase split is a DIFFERENT instrument: HIP-event brackets around 5 eager reps with a host barrier between reps, with the rank MAX taken independently per stage. Summing the three rank-max stages over-counts (max of sums <= sum of maxes) -- their sum exceeds production's p50 by ~7%.",
        "service_drain overlaps M7 and combine and must NOT be added into the stacked bar.",
        "dispatch_M0toM2 is NOT stamped in this build; only an instrument-mixed residual estimate is available.",
        "The two campaigns are identical replicates (same commit, same cfg, same seed, K0_SYNTH_ROUTE unset); rotations are pooled for the phase statistics and per-campaign medians are kept for drift inspection.",
    ],
    "resolution": {
        "stamp_sigma_prior_us": {"M6": 4.8, "plan": 4.7,
                                 "source": "prior aug11 work (exp_24/exp_28)"},
        "screen_vs_campaign": "screens resolve ~6.6% end-to-end; phase stamps resolve ~1%",
    },
    "gates": {
        "mok_gate_pass_counts": gates["mok_gate_pass_counts"],
        "mok_gate_failures": gates["mok_gate_fail"],
        "control_fails_values": sorted(gates["control_fails"]),
        "soak": gates["soak"],
        "poison_selftest": sorted([list(x) for x in gates["selftest"]]),
        "poison_survivor_values": sorted(gates["poison_survivors"]),
        "pperr_max": gates["pperr_max"],
        "all_green": (not gates["mok_gate_fail"]
                      and gates["control_fails"] == {"True"}
                      and all(s["pass"] == "True" and s["pperr"] == 0
                              and s["poison"] == 0 and s["completed"] == "600/600"
                              for s in gates["soak"])
                      and all(x[1] == "True" and x[2] == "57344" for x in gates["selftest"])
                      and sorted(gates["poison_survivors"]) == [0]
                      and gates["pperr_max"] == 0),
    },
    "peer_wait": {
        "instrument": "[MPS SPIN] chunk_poll -- M2's per-(source,chunk) dispatch poll, a running max over warmup + timed + all 600 soak epochs, never reset",
        "success_max": sorted({r["spin_success_max"] for r in rot}),
        "success_max_all_rotations": [r["spin_success_max"] for r in rot],
        "fail_max": sorted({r["spin_fail_max"] for r in rot}),
        "fail_max_all_rotations": [r["spin_fail_max"] for r in rot],
        "limit": rot[0].get("spin_limit") if rot else None,
        "interpretation": (
            "fail_max = {fmax} in every rotation and success_max = {smax} "
            "(max over {n} rotations), against a spin limit of 2,000,000. Each "
            "rotation's counter covers 500 warmup + 100 timed + 600 soak epochs, "
            "so a maximum of {smax} poll iteration(s) EVER means M2 essentially "
            "never waits on a peer chunk: dispatch peer wait is ~0 at this shape, "
            "reproducing exp_10's zero-peer-wait result at routing std=0."
        ).format(fmax=max(r["spin_fail_max"] for r in rot) if rot else None,
                 smax=max(r["spin_success_max"] for r in rot) if rot else None,
                 n=len(rot)),
    },
    "arms": {
        "production": arm_block("production", prod_phases, {
            "stage_sums": prod_stage_sums,
            "phase_instrument": "stage_profile.production (dispatch / gemm / combine), 3 coarse phases only",
            "phase_note": "the MPS stamp instrument does not exist for production (AITER/MoRI); this harness-native eager stage profile is the only per-phase signal it has, and it is NOT comparable tick-for-tick with the mps_mega device stamps",
        }),
        "pf6gm_mega": arm_block("pf6gm_mega", pf6gm_phases, {
            "phase_instrument": None,
            "phase_note": NO_PF6GM,
        }),
        "mps_mega": arm_block("mps_mega", mps_phases, {
            "phase_instrument": "[MPS TS] / [MPS TS SPLIT] / [MPS TS DELTA] device stamps, rank 0, final soak epoch",
            "stacked_bar_phases": ["dispatch_M0toM2", "plan_M3toM5", "M6_gemm1",
                                   "M7_gemm2_total", "combine_M8M9"],
            "stacked_bar_note": "these five partition the kernel; service_drain and interior_M3toM9 are diagnostics that overlap them",
        }),
    },
    "supplementary": {
        "pf_full_stage_profile_us": {
            "note": "the non-megakernel pf path measured in the same runs; n2_p1/n2_p2 are the payload-free GEMM-1/GEMM-2 donor bodies, used as the surcharge proxy",
            "maxrank": {k: agg(v) for k, v in pf_full_stage["maxrank"].items()},
            "rank0": {k: agg(v) for k, v in pf_full_stage["local"].items()},
        },
    },
    "derived": derived,
    "per_rotation_raw": rot,
}

os.makedirs(f"{HOME}/e33/out", exist_ok=True)
outp = f"{HOME}/e33/out/phase_stamps.json"
with open(outp, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=False)
print("wrote", outp, os.path.getsize(outp), "bytes")
print("rotations parsed:", len(rot), "| gates all_green:", doc["gates"]["all_green"])
print("p50 medians:", {a: doc["arms"][a]["p50_us"] for a in arm_names})
print("mps ratio_vs_production:", doc["arms"]["mps_mega"]["ratio_vs_production"])
print("identity residual (us):", derived["identity_check_us"]["max_abs_per_rotation_residual_us"])
for k, v in mps_phases.items():
    if v and v.get("us_rank0") is not None:
        print(f"  {k:22s} {v['us_rank0']:9.2f} +/- {v['stderr']} us (n={v['n']})")
print("coupled M7+combine:", derived["M7_plus_combine_coupled_block_us"]["value"],
      "+/-", derived["M7_plus_combine_coupled_block_us"]["stderr"],
      "pearson r =", derived["M7_plus_combine_coupled_block_us"]["pearson_r_M7_vs_combine"])
print("dispatch residual est:", derived["dispatch_M0toM2_residual_estimate_us"]["value"])
print("surcharge proxy A:", derived["m7_epilogue_surcharge_proxy_us"]["proxy_A_same_run_pf_full_n2_p2"]["value"])
print("production maxrank:", {k: v["us_rank_max"] for k, v in prod_phases.items()})
print("production rank0  :", {k: v["us_rank0"] for k, v in prod_phases.items()})
PY
RC=$?
echo "builder rc=$RC"

# ------------------------------------------------------------ raw bundle
BUN="$HOME/e33/out/e33_raw.tar.gz"
rm -f "$BUN"
FILES=()
for tag in "$@"; do
  for d in "$HOME/k0-mok-${tag}"/*/; do
    FILES+=("${d#$HOME/}")
  done
  for f in "$HOME/overnight-scratch/${tag}"_*.log "$HOME/overnight-scratch/screen_${tag}.csv" \
           "$HOME/e33/${tag}_driver.log" "$HOME/e33/${tag}.cfgs"; do
    [ -e "$f" ] && FILES+=("${f#$HOME/}")
  done
done
tar -czf "$BUN" -C "$HOME" "${FILES[@]}" 2>/dev/null
echo "bundle: $BUN  $(du -h "$BUN" | awk '{print $1}')"
exit 0
