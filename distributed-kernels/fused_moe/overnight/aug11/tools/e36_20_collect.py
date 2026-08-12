#!/usr/bin/env python3
"""exp_36 collector: turn every e36 run on the node into one plot-ready record.

Joins three sources per point:
  * <output_root>/summary.json   -> arm_p50_us medians (the only legitimate us)
  * $HOME/e36/scratch/<TAG>_<idx>_<slug>_<stamp>.log -> every gate line, the
    device stamps, [MPS SPIN], and the exp_36 shape echo
  * <output_root>/run1/k0pf_mok_synthetic_rank*.json -> the realised route,
    from which the measured destination-load imbalance is computed

Gates are re-derived here rather than trusted from the screen CSV: the CSV
parser deployed on the node predates the exp_32 poison fields and calls every
current [MPS SOAK] line MALFORMED.
"""
import json
import os
import re
import statistics
import sys
from glob import glob

HOME = os.path.expanduser("~")
SCRATCH = os.path.join(HOME, "e36", "scratch")


def parse_log(path):
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return {}
    out = {}
    m = re.search(r"\[exp_36 shape\] T=(\d+) T_LOC_MAX=(\d+) PADMAX=(\d+) "
                  r"MAXTOK=(\d+) MAXTOK_PROD=(\d+)", text)
    if m:
        out["shape"] = dict(T=int(m.group(1)), T_LOC_MAX=int(m.group(2)),
                            PADMAX=int(m.group(3)), MAXTOK=int(m.group(4)),
                            MAXTOK_PROD=int(m.group(5)))
    gates = {}
    for arm, a, r, p in re.findall(
            r"\[MOK GATE\] (\S+) max_abs=([0-9.eE+-]+) relative=([0-9.eE+-]+) pass=(\w+)", text):
        gates[arm] = dict(max_abs=float(a), relative=float(r), passed=(p == "True"))
    out["mok_gate"] = gates
    m = re.search(r"\[MARK\] control_fails=(\w+)", text)
    out["control_fails"] = (m.group(1) == "True") if m else None
    m = re.search(r"\[MPS SOAK\] completed=(\d+)/(\d+) pperr=(\d+)"
                  r"(?: poison=(-?\d+) poison_epoch=(-?\d+))? pass=(\w+)", text)
    if m:
        out["soak"] = dict(completed=int(m.group(1)), total=int(m.group(2)),
                           pperr=int(m.group(3)),
                           poison=(int(m.group(4)) if m.group(4) is not None else None),
                           passed=(m.group(6) == "True"))
    out["selftest"] = {a: (v == "True") for a, v in re.findall(
        r"\[POISON SELFTEST\] arm=(\S+) one_row_poisoned_fails=(\w+)", text)}
    out["poison_survivors_max"] = max(
        [int(v) for v in re.findall(r"\[POISON\] \S+ arm=\S+ survivors=(\d+)", text)] or [0])
    perr = [int(v) for v in re.findall(r"pperr=(\d+)", text)]
    out["pperr_max"] = max(perr) if perr else None
    m = re.search(r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+)", text)
    out["spin"] = dict(success_max=int(m.group(1)), fail_max=int(m.group(2))) if m else None
    d = re.findall(r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
                   r"servicedrain=(-?\d+) m2_to_end=(-?\d+)", text)
    s = re.findall(r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)", text)
    st = {}
    if d:
        k = ["planM6", "M7", "combine", "servicedrain", "m2_to_end"]
        st.update({n: int(v) / 100.0 for n, v in zip(k, d[-1])})
    if s:
        st.update({n: int(v) / 100.0 for n, v in zip(["plan_M3toM5", "M6"], s[-1])})
    out["stamps_us"] = st or None
    m = re.search(r"\[MOK SYNTHETIC EAGER\] status=(\S+) .*gate_ok=(\w+)", text)
    out["eager"] = dict(status=m.group(1), gate_ok=(m.group(2) == "True")) if m else None
    return out


def route_stats(run_dir):
    """Measured routing imbalance: the coefficient of variation of the number of
    (token, expert) assignments landing on each destination rank, aggregated
    over all eight source ranks.  This is the load skew the transport sees."""
    files = sorted(glob(os.path.join(run_dir, "k0pf_mok_synthetic_rank*.json")))
    if not files:
        return None
    dest = [0] * 8
    fanout = []
    emin, emax = [], []
    tokens = None
    for f in files:
        try:
            si = json.load(open(f)).get("synthetic_inputs", {})
        except Exception:
            continue
        # the realised-route block is nested one level down
        si = {**si, **(si.get("route") or {})}
        per = si.get("expert_assignments_per_destination_rank")
        if per:
            for i, v in enumerate(per):
                dest[i] += v
        if "fanout_mean" in si:
            fanout.append(si["fanout_mean"])
        if "expert_assignment_min" in si:
            emin.append(si["expert_assignment_min"])
            emax.append(si["expert_assignment_max"])
        sh = si.get("hidden_shape")
        if sh:
            tokens = sh[0]
    if not any(dest):
        return None
    mean = sum(dest) / len(dest)
    sd = statistics.pstdev(dest)
    return dict(tokens_per_rank=tokens,
                dest_load=dest,
                dest_load_cv=(sd / mean if mean else None),
                dest_load_min=min(dest), dest_load_max=max(dest),
                fanout_mean=(sum(fanout) / len(fanout) if fanout else None),
                expert_assignment_min=(min(emin) if emin else None),
                expert_assignment_max=(max(emax) if emax else None))


def cfg_from_csv(csv_path):
    """outdir basename -> (cfg string, tag)"""
    import csv as _csv
    out = {}
    try:
        rows = list(_csv.reader(open(csv_path)))
    except OSError:
        return out
    if not rows:
        return out
    head = rows[0]
    ix = {n: i for i, n in enumerate(head)}
    for r in rows[1:]:
        if len(r) <= ix["outdir"]:
            continue
        out[r[ix["outdir"]]] = dict(cfg=r[ix["cfg"]], iters=r[ix["iters"]],
                                    head=r[ix["head"]], src_rev=r[ix["src_rev"]],
                                    status=r[ix["status"]], note=r[-1])
    return out


def parse_cfg(cfg):
    d = {}
    for item in cfg.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                d[k.strip()] = int(v.strip(), 0)
            except ValueError:
                pass
    return d


DEPTH = {0: 8, 1: 4, 2: 16, 3: 32}


def throttle(g):
    if not (g & 0x20):
        return dict(enabled=False, depth=None)
    return dict(enabled=True, depth=DEPTH[(g & 0x300) >> 8])


def main():
    recs = []
    for csv_path in sorted(glob(os.path.join(SCRATCH, "screen_e36*.csv"))):
        tag = os.path.basename(csv_path)[len("screen_"):-len(".csv")]
        meta = cfg_from_csv(csv_path)
        for outdir, info in meta.items():
            root = os.path.join(HOME, f"k0-mok-{tag}", outdir)
            logs = glob(os.path.join(SCRATCH, f"{tag}_*{outdir.split('_', 1)[1]}.log"))
            log = logs[0] if logs else ""
            lg = parse_log(log) if log else {}
            summary = None
            sp = os.path.join(root, "summary.json")
            if os.path.exists(sp):
                try:
                    summary = json.load(open(sp))
                except Exception:
                    summary = None
            p50 = {}
            nvals = {}
            if summary:
                for arm in ("production", "pf6gm_mega", "mps_mega"):
                    try:
                        p50[arm] = float(summary["arm_p50_us"][arm]["median"])
                        nvals[arm] = len(summary["arm_p50_us"][arm].get("values", []))
                    except Exception:
                        p50[arm] = None
            cfgd = parse_cfg(info["cfg"])
            rs = route_stats(os.path.join(root, "run1"))
            gates = lg.get("mok_gate", {})
            soak = lg.get("soak") or {}
            green = bool(
                summary is not None
                and all(g.get("passed") for g in gates.values()) and gates
                and lg.get("control_fails") is True
                and soak.get("passed") is True
                and soak.get("completed") == 600 and soak.get("total") == 600
                and lg.get("pperr_max") == 0
                and lg.get("poison_survivors_max") == 0
                and lg.get("selftest")
                and all(lg["selftest"].values())
            )
            recs.append(dict(
                tag=tag, outdir=outdir, log=os.path.basename(log),
                cfg=info["cfg"], C=cfgd.get("C"), g=cfgd.get("g"),
                mode=cfgd.get("mode"), flush_rows=cfgd.get("flush_rows"),
                throttle=throttle(cfgd.get("g", 0)),
                iters=info["iters"], measurement=("campaign" if info["iters"].endswith("p5")
                                                  else "screen"),
                commit=info["head"], src_rev=info["src_rev"],
                shape=lg.get("shape"),
                p50_us=p50.get("mps_mega"),
                production_p50_us_same_run=p50.get("production"),
                pf6gm_p50_us_same_run=p50.get("pf6gm_mega"),
                ratio_vs_production=(p50.get("mps_mega") / p50["production"]
                                     if p50.get("mps_mega") and p50.get("production") else None),
                ratio_vs_pf6gm=(p50.get("mps_mega") / p50["pf6gm_mega"]
                                if p50.get("mps_mega") and p50.get("pf6gm_mega") else None),
                pf6gm_vs_production=(p50.get("pf6gm_mega") / p50["production"]
                                     if p50.get("pf6gm_mega") and p50.get("production") else None),
                n_processes=nvals.get("mps_mega"),
                spin_success_max=(lg.get("spin") or {}).get("success_max"),
                spin_fail_max=(lg.get("spin") or {}).get("fail_max"),
                phase_stamps_us=lg.get("stamps_us"),
                route=rs,
                gates=dict(mok_gate=gates, control_fails=lg.get("control_fails"),
                           soak=soak or None, pperr_max=lg.get("pperr_max"),
                           poison_survivors_max=lg.get("poison_survivors_max"),
                           poison_selftest=lg.get("selftest"), eager=lg.get("eager")),
                gates_green=green,
                csv_status=info["status"], note=info["note"],
            ))
    json.dump(recs, sys.stdout, indent=1, sort_keys=False)
    print()


if __name__ == "__main__":
    main()
