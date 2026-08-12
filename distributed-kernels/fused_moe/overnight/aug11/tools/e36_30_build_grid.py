#!/usr/bin/env python3
"""exp_36: turn collected.json into the plot-ready sensitivity_grid.json.

Every requested grid point appears exactly once, including the ones that could
not be measured: a point that the design refuses is a result, not a gap, so it
is carried as an explicit status row with the refusal text.
"""
import json
import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
COLLECTED = os.path.join(HOME, "e36", "collected.json")
OUT = os.path.join(HOME, "e36", "sensitivity_grid.json")

DEPTH_OF_G = {353: 4, 97: 8, 65: None, 1: None}

# every requested (T, std) cell, and what actually happened to it
T_REFUSAL = {
    512: dict(status="failed",
              error="e004pf_k0pf_ab.py:554 RuntimeError: pf3/pf4h/pf5/pf6 arms "
                    "are prefill-only; got T=512 (guard is T <= 512)",
              stage="host preflight, before any kernel launch"),
    1024: dict(status="failed",
               error="both megakernel arms fail the numeric output gate: "
                     "mps_mega leaves 7 of 8 ranks' outputs entirely unwritten "
                     "(7,340,032 = T*H poison survivors per rank), pf6gm_mega "
                     "returns wrong values (max_abs=1.71, relative=0.846). "
                     "production passes (max_abs=0.023). pperr=0 throughout.",
               stage="correctness gate, before timing"),
    2048: dict(status="failed",
               error="identical to T=1024: mps_mega leaves 7 of 8 ranks' outputs "
                     "unwritten (14,680,064 = T*H survivors), pf6gm_mega "
                     "max_abs=1.81 relative=0.873, production passes. pperr=0.",
               stage="correctness gate, before timing"),
}


def cfg_fields(cfg):
    d = {}
    for item in cfg.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                d[k.strip()] = int(v.strip(), 0)
            except ValueError:
                pass
    return d


def throttle_depth(g):
    if g is None or not (g & 0x20):
        return None
    return {0: 8, 1: 4, 2: 16, 3: 32}[(g & 0x300) >> 8]


def main():
    recs = json.load(open(COLLECTED))
    points = []

    # ---- measured points ----------------------------------------------------
    for r in recs:
        sh = r.get("shape") or {}
        T = sh.get("T")
        tag = r["tag"]
        if tag in ("e36c0",):            # driver bring-up, not a measurement
            continue
        cf = cfg_fields(r["cfg"])
        g = cf.get("g")
        rt = r.get("route") or {}
        st = r.get("phase_stamps_us") or {}
        seed = 2468 if tag in ("e36seedB", "e36campD") else 1234
        if tag == "e36route":
            status = "unreachable_by_harness"
            err = ("K0_SYNTH_ROUTE=skewed_hot raised on all 8 ranks: "
                   "'K0_SYNTH_ROUTE cannot be combined with "
                   "K0_INPUT_MODE=mok_synthetic' (e004pf_k0pf_ab.py:1866)")
        elif r["gates_green"]:
            status = "ok"
            err = None
        else:
            ref = T_REFUSAL.get(T, {})
            status = ref.get("status", "failed")
            err = ref.get("error", r.get("note") or "gates not green")
        points.append(dict(
            T=T,
            route_family="mok_synthetic router logits (torch.randn topk)",
            route_seed_base=seed,
            route_std_requested=None,
            route_std_measured=rt.get("dest_load_cv"),
            route_dest_load=rt.get("dest_load"),
            config=r["cfg"],
            C=cf.get("C"), g=g, mode=cf.get("mode"),
            flush_rows=cf.get("flush_rows"),
            throttle_enabled=(bool(g & 0x20) if g is not None else None),
            throttle_depth=throttle_depth(g),
            capacity_mode=("scaled" if (sh.get("MAXTOK") == T and T != 4096)
                           else "pinned_at_T4096"),
            shape=sh or None,
            status=status,
            error=err,
            measurement=r["measurement"],
            n_campaigns=(1 if r["measurement"] == "campaign" else 0),
            n_processes=r.get("n_processes"),
            p50_us=r["p50_us"],
            production_p50_us_same_run=r["production_p50_us_same_run"],
            pf6gm_p50_us_same_run=r["pf6gm_p50_us_same_run"],
            ratio_vs_production=r["ratio_vs_production"],
            ratio_vs_pf6gm=r["ratio_vs_pf6gm"],
            spin_success_max=r["spin_success_max"],
            spin_fail_max=r["spin_fail_max"],
            phase_stamps={"plan": st.get("plan_M3toM5"), "M6": st.get("M6"),
                          "M7": st.get("M7"), "combine": st.get("combine"),
                          "planM6": st.get("planM6"),
                          "servicedrain": st.get("servicedrain"),
                          "m2_to_end": st.get("m2_to_end")} if st else None,
            gates_green=r["gates_green"],
            gates=r["gates"],
            raw=dict(tag=tag, outdir=r["outdir"], log=r["log"]),
        ))

    # ---- requested-but-unreachable cells ------------------------------------
    for std in (0.032, 0.05):
        for T in (512, 1024, 2048, 4096):
            points.append(dict(
                T=T, route_family="skewed_hot (requested)",
                route_seed_base=0,
                route_std_requested=std, route_std_measured=None,
                config=None, C=None, g=None, mode=None, flush_rows=None,
                throttle_enabled=None, throttle_depth=None,
                capacity_mode=None, shape=None,
                status="unreachable_by_harness",
                error=("the routing-skew axis does not exist in this harness at "
                       "prefill: K0_SYNTH_ROUTE is rejected outright under "
                       "K0_INPUT_MODE=mok_synthetic (ab.py:1866, reproduced on "
                       "all 8 ranks), the synthetic-route families are "
                       "decode-only (WORLD=8,T=64,TOPK=8,E=32; ab.py:1948), and "
                       "synthetic_routes.py exposes no std parameter at all -- "
                       "skewed_hot is a fixed deterministic pattern, so "
                       "std=0.032/0.05 are not settable quantities here."),
                measurement=None, n_campaigns=0, n_processes=None,
                p50_us=None, production_p50_us_same_run=None,
                pf6gm_p50_us_same_run=None, ratio_vs_production=None,
                ratio_vs_pf6gm=None, spin_success_max=None, spin_fail_max=None,
                phase_stamps=None, gates_green=False, gates=None,
                raw=dict(tag="e36route", outdir=None, log=None),
            ))

    # ---- the C=0 placement rung that needs mode 14 --------------------------
    points.append(dict(
        T=4096, route_family="mok_synthetic router logits (torch.randn topk)",
        route_seed_base=1234, route_std_requested=0.05,
        route_std_measured=None,
        config="C=0,g=353,mode=12,flush_rows=16", C=0, g=353, mode=12,
        flush_rows=16, throttle_enabled=True, throttle_depth=4,
        capacity_mode="pinned_at_T4096", shape=None,
        status="requires_mode_14",
        error="C=0 is illegal in mode 12 (the validator rejects it); driving the "
              "CTA-dedication axis to zero needs mode 14, which was being built "
              "by another agent while this experiment ran. Not faked with a "
              "large-C proxy.",
        measurement=None, n_campaigns=0, n_processes=None, p50_us=None,
        production_p50_us_same_run=None, pf6gm_p50_us_same_run=None,
        ratio_vs_production=None, ratio_vs_pf6gm=None,
        spin_success_max=None, spin_fail_max=None, phase_stamps=None,
        gates_green=False, gates=None,
        raw=dict(tag=None, outdir=None, log=None),
    ))

    head = subprocess.run(
        ["git", "-C", os.path.join(HOME, "Distributed-HipKittens"),
         "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    doc = dict(
        schema_version="exp36.1",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        experiment="exp_36 sensitivity sweep (paper Fig 7 / Q5)",
        commit=head,
        node="gbt350-odcdh2-c05-1 (8x MI350X, gfx950)",
        harness="k0_fused_moe/benchmarks/mok_synthetic_prefill, arms "
                "production,pf6gm_mega,mps_mega",
        driver="$HOME/e36/rc_T.sh -- a sed-derived copy of run_campaign.sh whose "
               "ONLY delta is that the five shape env vars (K0_T, K0_T_LOC_MAX, "
               "K0_PADMAX, K0_MAXTOK, K0_MAXTOK_PROD) are parameterised. At "
               "T=4096 its defaults reproduce the hard-wired values exactly, so "
               "the T=4096 rows are directly comparable to every other "
               "measurement tonight.",
        units=dict(p50_us="microseconds, summary.json arm_p50_us[arm].median "
                          "(median over processes of the rank-max p50)",
                   phase_stamps="microseconds, device stamps from the FINAL "
                                "SOAK EPOCH (1 tick = 0.01 us); they do not sum "
                                "to p50_us",
                   route_std_measured="coefficient of variation of the number of "
                                      "(token,expert) assignments landing on each "
                                      "destination rank, summed over all 8 source "
                                      "ranks -- the load skew the transport sees"),
        caveats=[
            "The requested routing-std axis (0 / 0.032 / 0.05) is not a knob that "
            "exists in this harness at prefill. Rows for it carry "
            "status=unreachable_by_harness with the exact refusal.",
            "The requested T axis collapses to a single feasible point. T<=512 is "
            "refused by a host guard; T=1024 and T=2048 run but BOTH megakernel "
            "arms produce wrong output (mps_mega writes nothing on 7 of 8 ranks), "
            "so no timing from those shapes is admissible. T=4096 is the only "
            "shape where the kernel is correct.",
            "Screens (1 warmup / 1 timed / 1 process) rank against production "
            "only; 1 sigma on ratio_vs_production is 0.52% and the smallest "
            "callable single-screen delta is 2%. Only campaign rows "
            "(measurement=campaign, 5 processes, 500 warmup / 100 timed) are "
            "headline numbers.",
            "Every point carries timestamps=1 in K0_MPS_CFG so that all rows are "
            "internally paired; the ratchet numbers quoted elsewhere tonight were "
            "taken the same way.",
            "K0_MPS_SOAK_ITERS was left at 600 everywhere; no tolerance, shape, "
            "iteration count or warmup was altered to make a number look better.",
        ],
        points=points,
    )
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {OUT}: {len(points)} points "
          f"({sum(1 for p in points if p['status'] == 'ok')} measured green)")


if __name__ == "__main__":
    main()
