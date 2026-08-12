#!/usr/bin/env bash
# exp_35 step 8: build waterfall.json (paper Fig 4) from every collected
# campaign. Usage: bash e35_40_waterfall.sh TAG [TAG ...]
#
# Statistics discipline encoded here:
#  * the unit of analysis is the RUN (one process = one rotation, whose value is
#    already the MAX over the 8 ranks; the 8 ranks share one input and are NOT
#    independent). summarize.py has done the rank reduction; arm_p50_us.values
#    is therefore 5 run-level numbers per campaign.
#  * every ratio and every rung-to-rung delta is PAIRED WITHIN A CAMPAIGN, so
#    production and pf6gm_mega denominators are same-run by construction.
#  * a rung's headline number is the median over campaigns of the campaign
#    median; the per-campaign list is kept so the spread is visible.
set -uo pipefail
bash "$HOME/tools/e35_30_collect.sh" "$@" > /dev/null 2>&1 || true
python3 - "$@" <<'PY' > "$HOME/e35/waterfall.json"
import datetime, glob, json, os, re, statistics, sys

HOME = os.path.expanduser("~")
recs = json.load(open(f"{HOME}/e35/collected.json"))["records"]

CFG_C = "C=16,g=353,mode=12,flush_rows=16,timestamps=1"
CFG_B = "C=16,g=65,mode=12,flush_rows=16,timestamps=1"

def cfg_of(rec):
    m = re.match(r"\d+_(C\d+g\d+mode\d+flush_rows\d+timestamps\d+)_", rec["run_id"])
    if not m:
        return None
    s = m.group(1)
    p = re.match(r"C(\d+)g(\d+)mode(\d+)flush_rows(\d+)timestamps(\d+)", s)
    return (f"C={p.group(1)},g={p.group(2)},mode={p.group(3)},"
            f"flush_rows={p.group(4)},timestamps={p.group(5)}")

for r in recs:
    r["cfg"] = cfg_of(r)

def med(xs):
    return round(statistics.median(xs), 1) if xs else None

def sd(xs):
    return round(statistics.stdev(xs), 1) if len(xs) > 1 else None

def campaigns(cfg):
    return [r for r in recs if r["cfg"] == cfg and r["arm_p50_us"]]

def arm(rec, a, key="median"):
    v = rec["arm_p50_us"].get(a, {}).get(key)
    return round(v, 1) if isinstance(v, (int, float)) else v

def vals(rec, a):
    return [round(x, 1) for x in (rec["arm_p50_us"].get(a, {}).get("values") or [])]

def stamps(cs):
    """plan / M6 / M7 / combine, in us, medianed over every campaign of the rung."""
    def pull(path, key):
        out = []
        for c in cs:
            blk = (c.get(path) or {}).get(key)
            if blk:
                out.extend(blk["values_us"])
        return out
    plan = pull("ts_split", "plan_M3toM5")
    m6 = pull("ts_split", "M6")
    m7 = pull("ts_delta", "M7")
    comb = pull("ts_delta", "combine")
    drain = pull("ts_delta", "servicedrain")
    spin_ok = pull("spin", "success_max")
    spin_fail = pull("spin", "fail_max")
    return {
        "note": ("device MAX stamps from the FINAL SOAK epoch, 1 tick = 0.01 us; "
                 "they do NOT sum to arm_p50_us"),
        "plan_M3toM5_us": med(plan), "M6_us": med(m6), "M7_us": med(m7),
        "combine_us": med(comb), "service_drain_us": med(drain),
        "mps_spin_success_max": med(spin_ok), "mps_spin_fail_max": med(spin_fail),
        "n_samples": len(m7),
    }

def rung(rid, label, armname, cfg, mech, cs, prev_med):
    """armname is the arm whose p50 IS this rung; production is same-run paired."""
    per = []
    for c in cs:
        a, p, f = arm(c, armname), arm(c, "production"), arm(c, "pf6gm_mega")
        per.append({
            "run_id": c["run_id"],
            "arm_p50_us": a,
            "arm_p50_us_values": vals(c, armname),
            "production_p50_us_same_run": p,
            "pf6gm_mega_p50_us_same_run": f,
            "ratio_vs_production_same_run": round(a / p, 4) if a and p else None,
            "delta_us_vs_pf6gm_same_run": round(a - f, 1) if a and f else None,
            "gates_green": c["gates_green"],
            "pperr_max": c["pperr_max"],
            "soak": c["gates"]["soak"],
        })
    ms = [x["arm_p50_us"] for x in per if x["arm_p50_us"]]
    ps = [x["production_p50_us_same_run"] for x in per if x["production_p50_us_same_run"]]
    rs = [x["ratio_vs_production_same_run"] for x in per if x["ratio_vs_production_same_run"]]
    ds = [x["delta_us_vs_pf6gm_same_run"] for x in per if x["delta_us_vs_pf6gm_same_run"] is not None]
    m = med(ms)
    return {
        "rung_id": rid, "label": label, "arm": armname, "config": cfg,
        "mechanisms_present": mech,
        "status": "measured" if ms else "no_data",
        "n_campaigns": len(per),
        "arm_p50_us": {"per_campaign_median": ms, "median": m, "stdev": sd(ms),
                       "min": min(ms) if ms else None, "max": max(ms) if ms else None},
        "production_p50_us_same_run": {"per_campaign": ps, "median": med(ps),
                                       "stdev": sd(ps)},
        "ratio_vs_production": {"per_campaign": rs, "median": med_r(rs)},
        "delta_us_vs_pf6gm_mega_paired_same_run": {"per_campaign": ds, "median": med(ds),
                                                   "stdev": sd(ds)},
        "delta_us_vs_previous_rung": (round(m - prev_med, 1)
                                      if (m is not None and prev_med is not None) else None),
        "phase_stamps": stamps(cs),
        "gates_green": all(x["gates_green"] for x in per) if per else False,
    }

def med_r(rs):
    return round(statistics.median(rs), 4) if rs else None

cs_c, cs_b = campaigns(CFG_C), campaigns(CFG_B)
cs_all = cs_c + cs_b          # pf6gm_mega and production are arms in EVERY campaign

rungs = []
# ---- (a) homogeneous baseline: pf6gm_mega, measured in every campaign --------
ra = rung("a", "homogeneous baseline (no dedicated comm CTAs, no epilogue-carried payload)",
          "pf6gm_mega", "n/a (arm has no MPS config; K0_PF6GM_G=3)",
          [], cs_all, None)
ra["delta_us_vs_previous_rung"] = None
ra["ratio_vs_production"]["note"] = ("paired same-run against production in all "
                                     f"{len(cs_all)} campaigns")
# pf6gm_mega is a DIFFERENT kernel with no [MPS TS] instrumentation, so the MPS
# stamps found in these campaigns belong to the mps_mega arm, not to this rung.
ra["phase_stamps"] = None
ra["phase_stamps_note"] = ("none exist: the [MPS TS] rings are compiled into the mps "
                           "kernel only, so the homogeneous baseline has no phase "
                           "stamps in this harness")
ra["delta_us_vs_pf6gm_mega_paired_same_run"] = None
rungs.append(ra)
prev = ra["arm_p50_us"]["median"]

# ---- (b) + epilogue-carried payload, throttle DISABLED -----------------------
rb = rung("b", "+ epilogue-carried payload (mode 12 remote-accumulate), injection throttle DISABLED",
          "mps_mega", CFG_B,
          ["physical_g=1", "skip_dead_part_zero (0x40)", "throttle_enable=0 (0x20 CLEAR)",
           "depth_sel=0 (0x300 forced clear: config_is_valid rejects a selector "
           "without the enable bit)"],
          cs_b, prev)
rungs.append(rb)
prev = rb["arm_p50_us"]["median"]

# ---- (c) + injection bound (throttle depth 4) = the ratchet -------------------
rc = rung("c", "+ injection bound (remote-RMW throttle ON, depth 4) -- the current ratchet",
          "mps_mega", CFG_C,
          ["physical_g=1", "skip_dead_part_zero (0x40)", "throttle_enable=1 (0x20 SET)",
           "depth_sel=1 (0x300 -> depth 4)"],
          cs_c, prev)
rungs.append(rc)

# ---- (d)/(e): not built ------------------------------------------------------
rungs.append({
    "rung_id": "d",
    "label": "+ coarse arrival signals (mode 14: one grid barrier + 8 per-source arrival publishes)",
    "arm": "mps_mega", "config": None, "status": "pending_exp_34",
    "mechanisms_present": None, "n_campaigns": 0,
    "arm_p50_us": None, "production_p50_us_same_run": None,
    "ratio_vs_production": None, "delta_us_vs_previous_rung": None,
    "delta_us_vs_pf6gm_mega_paired_same_run": None,
    "phase_stamps": None, "gates_green": None,
    "blocked_by": "mode 14 is being built by another agent (exp_34); no binary existed at ca5b683f",
    "pre_registered_band_us": [5990, 6440],
})
rungs.append({
    "rung_id": "e",
    "label": "+ nc-major producer task order",
    "arm": "mps_mega", "config": None, "status": "not_built",
    "mechanisms_present": None, "n_campaigns": 0,
    "arm_p50_us": None, "production_p50_us_same_run": None,
    "ratio_vs_production": None, "delta_us_vs_previous_rung": None,
    "delta_us_vs_pf6gm_mega_paired_same_run": None,
    "phase_stamps": None, "gates_green": None,
    "blocked_by": "the producer task-order reorder (exp_30 companion) does not exist in any commit",
})

# ---- how much of the (a)->(c) gap each knob explains -------------------------
a_m, b_m, c_m = (rungs[0]["arm_p50_us"]["median"], rungs[1]["arm_p50_us"]["median"],
                 rungs[2]["arm_p50_us"]["median"])
total = round(c_m - a_m, 1)
attrib = {
    "total_gap_us_pf6gm_to_ratchet": total,
    "step_a_to_b_us": round(b_m - a_m, 1),
    "step_b_to_c_us": round(c_m - b_m, 1),
    "share_of_total_explained_by_payload_transport_alone_pct":
        round(100.0 * (b_m - a_m) / total, 1) if total else None,
    "share_of_total_explained_by_injection_bound_pct":
        round(100.0 * (c_m - b_m) / total, 1) if total else None,
    "reading": ("the epilogue-carried payload ALONE is a REGRESSION against the "
                "homogeneous baseline; the entire win, and then some, is carried by "
                "the injection bound -- i.e. by a scheduling decision, not by moving "
                "the payload and not by dedicating CTAs (C=16 is held FIXED across "
                "rungs b and c)"),
    "cta_dedication_axis": ("cannot be driven to zero inside mode 12: "
                            "config_is_valid rejects C=0 for every mode in "
                            "mode_is_stream, which includes 12 and 13 "
                            "(moe_mps_adapter.cuh:373). The only C=0 point available "
                            "is rung (a) itself, a different kernel. Isolating the "
                            "dedication axis needs mode 14 (exp_37, C in {0,8,16})."),
}

out = {
    "schema_version": "exp35.waterfall.1",
    "experiment": "exp_35 knob waterfall (paper Fig 4 / Q1)",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": "ca5b683f",
    "commit_full": "ca5b683f420e4985e8d2a790fe086419ab6730f1",
    "node": "gbt350-odcdh2-c05-1 (8x MI350X, gfx950)",
    "harness": {
        "driver": "benchmarks/mok_synthetic_prefill/run_campaign.sh via tools/screen.sh",
        "arms": "production,pf6gm_mega,mps_mega",
        "warmup_iters": 500, "timed_iters": 100, "rotations_per_campaign": 5,
        "soak_iters": 600, "T": 4096, "route": "default (std=0)", "seed": 0,
        "src_rev": 26, "K0P6_MPS_ASCALE_TM": 1,
    },
    "statistic": ("arm_p50_us[arm].median = median over the campaign's 5 runs of that "
                  "run's MAX-over-8-ranks p50. Clustering is by RUN, never by "
                  "(run, rank): the 8 ranks of a run share one input."),
    "g_field_bit_layout": {
        "source": "moe_mps_adapter.cuh:55-105 (packing), :224-265 (bit table), :338-375 (validator)",
        "packing": ("g is 16 bits SPLIT across the packed uint64: low byte at word[8:16), "
                    "high byte at word[34:42). The split exists because passing g>0xFF to "
                    "the old encoder overflowed bit 16 into the MODE field and still "
                    "validated (g=0x121 -> mode|=1, decoded back as g=0x21)."),
        "bits": {
            "0x000F": "physical g (must be 1 in modes 12/13)",
            "0x0010": "dual-write lost-update detector",
            "0x0020": "epilogue remote-RMW throttle ENABLE  <-- THE THROTTLE SWITCH",
            "0x0040": "exp_24 A: skip the dead `part` zero-fill in M5",
            "0x0080": "reserved (rejected)",
            "0x0300": "throttle DEPTH select: 00->8, 01->4, 10->16, 11->32",
            "0xFC00": "reserved (rejected)",
        },
        "unthrottled_encoding": ("clear 0x20. There is NO 'disabled' code point inside the "
                                 "0x300 depth field; disabling is the enable bit, and "
                                 "config_is_valid (:356-359) REJECTS a nonzero depth "
                                 "selector when the enable bit is clear. So the unique "
                                 "legal 'throttle bits only' neighbour of g=353 (0x161) "
                                 "is g=65 (0x041), not g=321 (0x141)."),
        "rung_c_g": {"decimal": 353, "hex": "0x161", "physical_g": 1, "detect": 0,
                     "throttle_enable": 1, "skip_part_zero": 1, "depth_sel": 1,
                     "depth": 4},
        "rung_b_g": {"decimal": 65, "hex": "0x041", "physical_g": 1, "detect": 0,
                     "throttle_enable": 0, "skip_part_zero": 1, "depth_sel": 0,
                     "depth": "n/a (throttle off)"},
        "empirical_proof": {
            "1_descriptor_dump": {
                "method": ("K0_MPS_DESC_DUMP=1 dumps the mps descriptor as the host handed "
                           "it to the kernel; slot K0P6_D_MPS_CFG=62 is the packed word "
                           "that k0pf6gm_device_tile_mps.hip:702 feeds to decode_config()"),
                "rung_c_word": "26039050512 (0x6100c6110)",
                "rung_b_word": "8859173136 (0x2100c4110)",
                "predicted_by_encode_config": "identical, bit for bit, for both",
                "mode_field_readback": 12,
                "why_it_matters": ("mode==12 in the readback is the direct check that the "
                                   "historical g-overflow-into-mode bug is NOT present"),
            },
            "2_mechanism_signature": {
                "method": "timestamps=1 phase stamps, throttle ON vs OFF",
                "M7_us_throttled_vs_unthrottled": "see rungs b/c phase_stamps",
                "independent_expectation": ("exp_24 measured depth 16 and 32 returning M7 "
                                            "to the UNTHROTTLED cost at +462 and +370 us "
                                            "vs the depth-4 ratchet; a genuinely "
                                            "unthrottled config must land in that band"),
                "one_variable_check": ("plan_M3toM5 and M6 must be UNCHANGED between the "
                                       "two rungs -- the throttle lives only in the "
                                       "phase-2 epilogue"),
            },
            "3_fail_closed_negative_control": {
                "config": "C=16,g=321,mode=12,flush_rows=16,timestamps=1",
                "expectation": "REJECTED by config_is_valid (depth selector, enable clear)",
                "observed": ("FAIL:rc23 / missing-rank-json, every gate VOID -- the device "
                             "guard at .hip:719 raised K0P6_MPS_ERR_CONFIG (1<<28) and "
                             "returned before producing output"),
                "conclusion": ("g=321 is not a legal rung, which is exactly why rung (b) "
                               "is g=65"),
                "note": "EXPECTED failure. Not a rung, not a candidate regression.",
            },
        },
    },
    "attribution": attrib,
    "caveats": [
        "Rung (a) -> (b) is NOT a single-variable step and cannot be made one: the "
        "homogeneous baseline is a different kernel (pf6gm_mega) with no MPS protocol at "
        "all, so that step bundles the mode-12 remote-accumulate transport, the C=16 "
        "service pool, and the per-row completion protocol. Rung (b) -> (c) IS strictly "
        "single-variable: same arm, same commit, same C, same flush_rows, one g bit.",
        "The throttle is a COMPILE-TIME specialization (four s_waitcnt vmcnt(N) "
        "instantiations, moe_mps_adapter.cuh:260-261), so rungs (b) and (c) resolve to "
        "different .hsaco builds (be7b189d458d vs 5635a2f2a370) from the SAME pinned "
        "source at ca5b683f. That is the mechanism, not a confound.",
        "C=0 is illegal in mode 12 (moe_mps_adapter.cuh:373), so this waterfall cannot "
        "separate 'dedicated comm CTAs' from 'mode-12 transport' inside a single arm. "
        "What it CAN show is that with C held fixed at 16, one scheduling bit moves "
        "end-to-end by more than the entire (a)->(c) gap.",
        "Phase stamps are device MAX stamps from the final soak epoch, not from a timed "
        "iteration; they are a phase story, not an additive budget.",
        "Rungs (d) and (e) are unmeasured: no binary for either existed at ca5b683f.",
        "out is not bit-reproducible in ANY arm (see STATUS.md at this commit); "
        "correctness is the [MOK GATE] tolerance gate plus the NaN-poison detector, "
        "not a bit compare.",
    ],
    "rungs": rungs,
    "campaigns_raw": [
        {"run_id": r["run_id"], "cfg": r["cfg"], "gates_green": r["gates_green"],
         "pperr_max": r["pperr_max"],
         "arm_p50_us": {a: {"median": arm(r, a), "values": vals(r, a)}
                        for a in sorted(r["arm_p50_us"])},
         "gate_lines": r["gates"],
         "ts_delta": r["ts_delta"], "ts_split": r["ts_split"], "spin": r["spin"]}
        for r in sorted(recs, key=lambda x: x["run_id"])
    ],
}
print(json.dumps(out, indent=1))
PY
echo "===waterfall.json written: $(wc -c < "$HOME/e35/waterfall.json") bytes===" >&2
python3 -c "
import json,sys
d=json.load(open('$HOME/e35/waterfall.json'))
print('RUNG TABLE')
for r in d['rungs']:
    if r.get('arm_p50_us'):
        print(' %-2s %-9s n=%d p50=%8.1f  prod=%8.1f  ratio=%.4f  d_prev=%s  M7=%s' % (
            r['rung_id'], r['arm'], r['n_campaigns'], r['arm_p50_us']['median'],
            r['production_p50_us_same_run']['median'],
            r['ratio_vs_production']['median'], r['delta_us_vs_previous_rung'],
            (r['phase_stamps'] or {}).get('M7_us')))
        print('      per_campaign=%s  gates_green=%s' % (r['arm_p50_us']['per_campaign_median'], r['gates_green']))
    else:
        print(' %-2s %-9s %s' % (r['rung_id'], r['arm'], r['status']))
print('ATTRIB', json.dumps(d['attribution'], indent=1)[:700])
" >&2
