#!/usr/bin/env bash
# exp_34: build mode14_arms.json -- one record per ARM (mode, C, g, stamps),
# aggregated over its campaigns. Clustering is by RUN: a campaign's value is the
# median over its 5 processes of the rank-max p50 (that is what summary.json's
# arm_p50_us[arm].values already are), and an arm's value is the median over its
# campaigns. Usage: bash e34_51_collect.sh TAG [TAG2 ...]
set -uo pipefail
OUT=$HOME/e34/mode14_arms.json
python3 - "$@" <<'PY' > "$OUT"
import glob, json, os, re, statistics, sys

HOME = os.path.expanduser("~")
tags = sys.argv[1:] or ["e34c1"]

def med(x):
    x = [v for v in x if v is not None]
    return round(statistics.median(x), 1) if x else None

campaigns = []
for tag in tags:
    for log in sorted(glob.glob(f"{HOME}/overnight-scratch/{tag}_*.log")):
        base = os.path.basename(log)[:-4]
        suffix = base[len(tag) + 1:]
        outdir = f"{HOME}/k0-mok-{tag}/{suffix}"
        text = open(log, errors="ignore").read()
        cfgm = re.search(r"K0_MPS_CFG=(\S+)", text)
        cfg = cfgm.group(1) if cfgm else ""
        def fld(k, d=None):
            m = re.search(rf"(?:^|,){k}=(\d+)", cfg)
            return int(m.group(1)) if m else d
        c = dict(tag=tag, run_id=suffix, log=os.path.basename(log), cfg=cfg,
                 mode=fld("mode"), C=fld("C"), g=fld("g"),
                 stamps=fld("timestamps", 0))
        sj = os.path.join(outdir, "summary.json")
        c["arms"] = {}
        if os.path.exists(sj):
            s = json.load(open(sj))
            for arm, st in (s.get("arm_p50_us") or {}).items():
                c["arms"][arm] = {"median": st.get("median"),
                                  "per_run_values": st.get("values")}
        # gates
        perrs = [int(v) for v in re.findall(r"pperr=(\d+)", text)]
        surv = [int(v) for v in re.findall(r"survivors=(\d+)", text)]
        gate = re.findall(r"\[MOK GATE\] mps_mega max_abs=\S+ relative=\S+ pass=(\w+)", text)
        ctl = re.findall(r"\[MARK\] control_fails=(\w+)", text)
        soak = re.findall(r"\[MPS SOAK\] completed=(\d+)/(\d+) pperr=(\d+).*?pass=(\w+)", text)
        stdead = [a for a, v in re.findall(
            r"\[POISON SELFTEST\] arm=(\S+) one_row_poisoned_fails=(\w+)", text) if v != "True"]
        c["pperr_max"] = max(perrs) if perrs else None
        c["poison_survivors_max"] = max(surv) if surv else None
        c["soak_epochs"] = sorted({f"{a}/{b}" for a, b, _, _ in soak})
        c["gates_green"] = bool(
            gate and all(v == "True" for v in gate)
            and ctl and all(v == "True" for v in ctl)
            and soak and all(d == "True" for *_, d in soak)
            and c["pperr_max"] == 0 and not stdead
            and (c["poison_survivors_max"] in (0, None)))
        c["gate_lines"] = sorted({l.strip() for l in re.findall(
            r"^.*\[(?:MOK GATE\] mps_mega|MPS SOAK\]|MARK\] control_fails|POISON SELFTEST\]).*$",
            text, re.M)})
        # stamps (ticks -> us; MAX stamps from the final soak epoch, never summed
        # against arm_p50_us)
        def stamp(pat, keys):
            hits = re.findall(pat, text)
            if not hits:
                return None
            cols = list(zip(*[[int(x) for x in h] for h in hits]))
            return {k: med([v / 100.0 for v in col]) for k, col in zip(keys, cols)}
        c["ts_delta"] = stamp(
            r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
            r"servicedrain=(-?\d+) m2_to_end=(-?\d+)",
            ["planM6", "M7", "combine", "servicedrain", "m2_to_end"])
        c["ts_split"] = stamp(r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)",
                              ["plan_M3toM5", "M6"])
        c["spin"] = stamp(r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+)",
                          ["success_max", "fail_max"])
        campaigns.append(c)

# ---- group into arms -------------------------------------------------------
def arm_id(c):
    return f"mode{c['mode']}_C{c['C']}_g{c['g']}_ts{c['stamps']}"

arms = {}
for c in campaigns:
    arms.setdefault(arm_id(c), []).append(c)

def mps(c):  return (c["arms"].get("mps_mega") or {}).get("median")
def prod(c): return (c["arms"].get("production") or {}).get("median")
def pf6(c):  return (c["arms"].get("pf6gm_mega") or {}).get("median")

records = []
for aid, cs in sorted(arms.items()):
    ok = [c for c in cs if c["gates_green"] and mps(c) is not None]
    rec = {
        "arm_id": aid,
        "mode": cs[0]["mode"], "C": cs[0]["C"], "g": cs[0]["g"],
        "stamps": cs[0]["stamps"],
        "status": "OK" if ok else "NO_GREEN_CAMPAIGN",
        "campaigns": [{
            "tag": c["tag"], "run_id": c["run_id"],
            "mps_mega_p50_us": mps(c), "production_p50_us": prod(c),
            "pf6gm_mega_p50_us": pf6(c),
            "ratio_vs_production": (round(mps(c) / prod(c), 4)
                                    if mps(c) and prod(c) else None),
            "mps_per_run_values_us": (c["arms"].get("mps_mega") or {}).get("per_run_values"),
            "production_per_run_values_us": (c["arms"].get("production") or {}).get("per_run_values"),
            "gates_green": c["gates_green"], "pperr_max": c["pperr_max"],
            "soak_epochs": c["soak_epochs"],
        } for c in cs],
        "p50_median": med([mps(c) for c in ok]),
        "production_p50_same_run": med([prod(c) for c in ok]),
        "ratio_vs_production": med([mps(c) / prod(c) * 1000 for c in ok
                                    if mps(c) and prod(c)]),
        "phase_stamps": {
            "ts_delta": {k: med([c["ts_delta"][k] for c in ok if c.get("ts_delta")])
                         for k in ["planM6", "M7", "combine", "servicedrain", "m2_to_end"]}
                         if any(c.get("ts_delta") for c in ok) else None,
            "ts_split": {k: med([c["ts_split"][k] for c in ok if c.get("ts_split")])
                         for k in ["plan_M3toM5", "M6"]}
                         if any(c.get("ts_split") for c in ok) else None,
            "spin": {k: med([c["spin"][k] for c in ok if c.get("spin")])
                     for k in ["success_max", "fail_max"]}
                     if any(c.get("spin") for c in ok) else None,
        },
        "gates_green": bool(ok) and all(c["gates_green"] for c in cs),
        "n_campaigns": len(ok),
        "n_campaigns_attempted": len(cs),
        "gate_lines": sorted({l for c in cs for l in c["gate_lines"]})[:8],
    }
    if rec["ratio_vs_production"] is not None:
        rec["ratio_vs_production"] = round(rec["ratio_vs_production"] / 1000.0, 4)
    records.append(rec)

# ---- delta vs the same-stamps mode-12 control ------------------------------
ctrl = {r["stamps"]: r for r in records if r["mode"] == 12}
for r in records:
    c = ctrl.get(r["stamps"])
    if c and r["p50_median"] is not None and c["p50_median"] is not None:
        r["delta_vs_mode12_control"] = round(r["p50_median"] - c["p50_median"], 1)
        r["mode12_control_arm_id"] = c["arm_id"]
        r["mode12_control_p50_us"] = c["p50_median"]
    else:
        r["delta_vs_mode12_control"] = None
        r["mode12_control_arm_id"] = c["arm_id"] if c else None

print(json.dumps({"experiment": "exp_34_mode14",
                  "pin": "291dfa08",
                  "src_rev": 28,
                  "harness": "mok_synthetic_prefill run_campaign.sh, 5 rotations, "
                             "500 warmup / 100 timed, K0_MPS_SOAK_ITERS=600",
                  "clustering": "campaign value = median over 5 processes of the "
                                "rank-max p50; arm value = median over campaigns",
                  "arms": records}, indent=1))
PY
echo "===WROTE $OUT ($(wc -c < "$OUT") bytes)==="
python3 - "$OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"{'arm_id':28s} {'n':>2s} {'p50':>8s} {'prod':>8s} {'ratio':>7s} {'d_vs_m12':>9s} green")
for r in d["arms"]:
    print(f"{r['arm_id']:28s} {r['n_campaigns']:2d} "
          f"{(r['p50_median'] or 0):8.1f} {(r['production_p50_same_run'] or 0):8.1f} "
          f"{(r['ratio_vs_production'] or 0):7.4f} "
          f"{(r['delta_vs_mode12_control'] if r['delta_vs_mode12_control'] is not None else 0):9.1f} "
          f"{r['gates_green']}")
PY
