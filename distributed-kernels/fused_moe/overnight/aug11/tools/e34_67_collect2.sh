#!/usr/bin/env bash
# exp_34 collector v2 -> mode14_arms.json.
#
# Authoritative source is screen_<tag>.csv (it carries cfg, status, gates, pperr,
# soak, the three arm p50s and the phase stamps); per-PROCESS values come from
# each campaign's summary.json so the clustering is by RUN, never by (run, rank).
# Arms are keyed by REVISION as well as (mode, C, g, stamps), because the rev-26
# control is part of tonight's evidence.
# Usage: bash e34_67_collect2.sh [TAG ...]   (defaults to tonight's tags)
set -uo pipefail
OUT=$HOME/e34/mode14_arms.json
TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=(e34c1 e34d e34r26 e34e e34f)
python3 - "${TAGS[@]}" <<'PY' > "$OUT"
import csv, glob, json, os, re, statistics, sys

HOME = os.path.expanduser("~")
tags = sys.argv[1:]

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None

# Mode 14 never enters the service drain, so K0P6_MPS_TS_DRAIN is never stored
# and the cell reads as its never-written sentinel (a huge negative after the
# tick conversion). Report it as unwritten rather than as a duration -- that
# absence IS the service-pool degeneration evidence.
DRAIN_UNWRITTEN = "UNWRITTEN: no CTA entered the drain (mode 14 skips it)"

def sane(v):
    return v if (v is not None and -1e5 < v < 1e6) else None

def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

campaigns = []
for tag in tags:
    csvp = f"{HOME}/overnight-scratch/screen_{tag}.csv"
    if not os.path.exists(csvp):
        continue
    for row in csv.DictReader(open(csvp)):
        cfg = row.get("cfg") or ""
        def fld(k, d=None):
            m = re.search(rf"(?:^|,){k}=(\d+)", cfg)
            return int(m.group(1)) if m else d
        outdir = row.get("outdir") or ""
        root = f"{HOME}/k0-mok-{tag}/{outdir}"
        c = dict(tag=tag, run_id=outdir, cfg=cfg, status=row.get("status"),
                 head=row.get("head"), src_rev=int(row.get("src_rev") or 0),
                 mode=fld("mode"), C=fld("C"), g=fld("g"),
                 stamps=fld("timestamps", 0), iters=row.get("iters"),
                 mps=fnum(row.get("mps_us")), prod=fnum(row.get("prod_us")),
                 pf6=fnum(row.get("pf6gm_us")),
                 gate_pass=(row.get("gate_pass") == "True"),
                 control_fails=(row.get("control_fails") == "True"),
                 soak=(row.get("soak") == "True"),
                 soak_epochs=row.get("soak_epochs"),
                 pperr=row.get("pperr"), hsaco=row.get("mps_hsaco"),
                 stamps_us={k: sane(fnum(row.get(v))) for k, v in
                            (("planM6", "ts_planM6_us"), ("M7", "ts_M7_us"),
                             ("combine", "ts_combine_us"),
                             ("servicedrain", "ts_servicedrain_us"),
                             ("m2_to_end", "ts_m2_to_end_us"),
                             ("plan_M3toM5", "ts_planM3toM5_us"),
                             ("M6", "ts_M6_us"))},
                 servicedrain_raw=fnum(row.get("ts_servicedrain_us")),
                 spin={"success_max": fnum(row.get("spin_ok_max")),
                       "fail_max": fnum(row.get("spin_fail_max"))})
        c["gates_green"] = bool(c["gate_pass"] and c["control_fails"] and c["soak"]
                                and c["pperr"] in ("0", 0)
                                and c["status"] == "OK")
        # per-RUN values (one per campaign process) out of summary.json
        sj = os.path.join(root, "summary.json")
        c["per_run"] = {}
        if os.path.exists(sj):
            s = json.load(open(sj))
            for arm, st in (s.get("arm_p50_us") or {}).items():
                c["per_run"][arm] = st.get("values")
            c["n_runs"] = len(s.get("runs") or [])
        # literal gate lines
        log = glob.glob(f"{HOME}/overnight-scratch/{tag}_*{outdir.split('_',1)[-1]}.log")
        c["gate_lines"] = []
        if log:
            t = open(log[0], errors="ignore").read()
            c["gate_lines"] = sorted({l.strip() for l in re.findall(
                r"^.*\[(?:MOK GATE\] mps_mega|MPS SOAK\]|MARK\] control_fails|"
                r"POISON SELFTEST\]|POISON\] post_timing arm=mps_mega).*$", t, re.M)})
        campaigns.append(c)

def arm_id(c):
    return f"rev{c['src_rev']}_mode{c['mode']}_C{c['C']}_g{c['g']}_ts{c['stamps']}"

arms = {}
for c in campaigns:
    arms.setdefault(arm_id(c), []).append(c)

# g bitfield decode, so the JSON is self-describing for the plot
def g_bits(g):
    if g is None:
        return None
    return {"physical_g": g & 0x0F, "detect_dual": bool(g & 0x10),
            "throttle": bool(g & 0x20), "skip_part_zero": bool(g & 0x40),
            "coarse_keep_drain": bool(g & 0x80),
            "throttle_depth_sel": (g & 0x300) >> 8}

records = []
for aid, cs in sorted(arms.items()):
    ok = [c for c in cs if c["gates_green"] and c["mps"] is not None]
    rec = {
        "arm_id": aid, "src_rev": cs[0]["src_rev"], "head": cs[0]["head"],
        "mode": cs[0]["mode"], "C": cs[0]["C"], "g": cs[0]["g"],
        "g_decode": g_bits(cs[0]["g"]), "stamps": cs[0]["stamps"],
        "status": "OK" if ok else "NO_GREEN_CAMPAIGN",
        "campaigns": [{
            "tag": c["tag"], "run_id": c["run_id"], "iters": c["iters"],
            "mps_mega_p50_us": c["mps"], "production_p50_us": c["prod"],
            "pf6gm_mega_p50_us": c["pf6"],
            "ratio_vs_production": (round(c["mps"] / c["prod"], 4)
                                    if c["mps"] and c["prod"] else None),
            "mps_per_run_values_us": (c["per_run"] or {}).get("mps_mega"),
            "production_per_run_values_us": (c["per_run"] or {}).get("production"),
            "gates_green": c["gates_green"], "pperr": c["pperr"],
            "soak_epochs": c["soak_epochs"], "phase_stamps": c["stamps_us"],
            "hsaco": c["hsaco"],
        } for c in cs],
        "p50_median": med([c["mps"] for c in ok]),
        "production_p50_same_run": med([c["prod"] for c in ok]),
        "pf6gm_p50_same_run": med([c["pf6"] for c in ok]),
        "phase_stamps": {k: med([c["stamps_us"][k] for c in ok])
                         for k in ok[0]["stamps_us"]} if ok else None,
        "servicedrain_status": (
            DRAIN_UNWRITTEN if (cs[0]["mode"] == 14 and cs[0]["stamps"] == 1)
            else ("written" if cs[0]["stamps"] == 1 else "stamps off")),
        "servicedrain_raw_us": [c["servicedrain_raw"] for c in cs],
        "spin": {k: med([c["spin"][k] for c in ok]) for k in ("success_max", "fail_max")}
                 if ok else None,
        "gates_green": bool(ok) and all(c["gates_green"] for c in cs),
        "n_campaigns": len(ok), "n_campaigns_attempted": len(cs),
        "gate_lines": sorted({l for c in cs for l in c["gate_lines"]})[:10],
    }
    if rec["p50_median"] and rec["production_p50_same_run"]:
        rec["ratio_vs_production"] = round(
            rec["p50_median"] / rec["production_p50_same_run"], 4)
    else:
        rec["ratio_vs_production"] = None
    records.append(rec)

# deltas against the mode-12 control of the SAME stamps class and SAME revision
ctrl = {(r["src_rev"], r["stamps"]): r for r in records
        if r["mode"] == 12 and r["g"] == 353}
for r in records:
    k = (r["src_rev"], r["stamps"])
    c = ctrl.get(k)
    r["mode12_control_arm_id"] = c["arm_id"] if c else None
    r["mode12_control_p50_us"] = c["p50_median"] if c else None
    r["delta_vs_mode12_control"] = (
        round(r["p50_median"] - c["p50_median"], 1)
        if c and r["p50_median"] and c["p50_median"] else None)
    # and against the TRUE ratchet: mode 12 g=353 at rev 26, same stamps class
    t = ctrl.get((26, r["stamps"]))
    r["ratchet_rev26_p50_us"] = t["p50_median"] if t else None
    r["delta_vs_ratchet_rev26"] = (
        round(r["p50_median"] - t["p50_median"], 1)
        if t and r["p50_median"] and t["p50_median"] else None)

print(json.dumps({
    "experiment": "exp_34_mode14",
    "pin": "291dfa08", "src_rev": 28,
    "band_pre_registered_us": [5990, 6440],
    "falsification_threshold_us": 6568,
    "harness": "mok_synthetic_prefill run_campaign.sh, 5 rotations, "
               "500 warmup / 100 timed, K0_MPS_SOAK_ITERS=600, T=4096",
    "clustering": "campaign value = median over the 5 processes of the rank-max "
                  "p50 (summary.json arm_p50_us.values); arm value = median over "
                  "campaigns. Never clustered by (run, rank).",
    "stamps_note": "timestamps=1 is NOT free; stamps-on and stamps-off arms are "
                   "separate records and are never compared to each other.",
    "arms": records}, indent=1))
PY
echo "===WROTE $OUT ($(wc -c < "$OUT") bytes)==="
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
h = f"{'arm_id':34s} {'n':>2s} {'p50':>8s} {'prod':>8s} {'ratio':>6s} {'dM12':>7s} {'dRat26':>7s} {'M7':>7s} {'drain':>7s} green"
print(h)
for r in d["arms"]:
    f = lambda v: (f"{v:8.1f}" if isinstance(v, (int, float)) else f"{'-':>8s}")
    ps = r.get("phase_stamps") or {}
    print(f"{r['arm_id']:34s} {r['n_campaigns']:2d} {f(r['p50_median'])} "
          f"{f(r['production_p50_same_run'])} "
          f"{(f'{r['ratio_vs_production']:.4f}' if r['ratio_vs_production'] else '-'):>6s} "
          f"{(f'{r['delta_vs_mode12_control']:+.1f}' if r['delta_vs_mode12_control'] is not None else '-'):>7s} "
          f"{(f'{r['delta_vs_ratchet_rev26']:+.1f}' if r['delta_vs_ratchet_rev26'] is not None else '-'):>7s} "
          f"{(f'{ps.get('M7'):.0f}' if ps.get('M7') else '-'):>7s} "
          f"{(f'{ps.get('servicedrain'):.0f}' if ps.get('servicedrain') else '-'):>7s} "
          f"{r['gates_green']}")
PY
