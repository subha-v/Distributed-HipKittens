#!/usr/bin/env python3
"""exp_37 collector: one consolidated JSON per campaign, run-level granularity.

Walks $HOME/k0-mok-e37*/*/summary.json, pairs each campaign with its screen.sh
log (the log name is derivable from the output-dir basename), and emits
$HOME/e37/e37_raw.json.

The unit that matters for statistics is the RUN (one of the 5 rotations): each
run contributes ONE rank-max-aligned p50 per arm, so per-run values are kept
verbatim and never flattened across ranks.
"""
import glob
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
SCRATCH = os.path.join(HOME, "overnight-scratch")
OUT = os.path.join(HOME, "e37", "e37_raw.json")


def gates_from_log(path):
    try:
        with open(path, errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return {"log": None}
    g = {}
    m = re.search(r"\[MOK GATE\] mps_mega max_abs=(\S+) relative=(\S+) pass=(\w+)", text)
    g["mok_gate"] = m.group(0) if m else None
    g["mok_gate_pass"] = m.group(3) if m else "VOID"
    m = re.search(r"\[MARK\] control_fails=(\w+)", text)
    g["control_fails"] = m.group(1) if m else "VOID"
    perr = [int(v) for v in re.findall(r"pperr=(\d+)", text)]
    g["pperr_max"] = max(perr) if perr else None
    soak = re.findall(r"\[MPS SOAK\][^\n]*", text)
    g["soak_lines"] = sorted(set(soak))[:3]
    m = re.search(r"\[MPS SOAK\] completed=(\d+)/(\d+)[^\n]*?pass=(\w+)", text)
    g["soak_epochs"] = f"{m.group(1)}/{m.group(2)}" if m else None
    g["soak_pass"] = m.group(3) if m else "VOID"
    st = re.findall(r"\[POISON SELFTEST\][^\n]*", text)
    g["poison_selftest_lines"] = sorted(set(st))[:3]
    g["poison_selftest_all_true"] = (
        bool(st)
        and all(v == "True" for v in re.findall(
            r"\[POISON SELFTEST\] arm=\S+ one_row_poisoned_fails=(\w+)", text))
    )
    nf = re.findall(r"one_row_poisoned_fails=\w+ nonfinite=(\d+)", text)
    g["poison_nonfinite"] = sorted(set(int(v) for v in nf))
    surv = re.findall(r"\[POISON\] (\S+) arm=(\S+) survivors=(\d+)", text)
    g["poison_survivors_max"] = max((int(s[2]) for s in surv), default=None)
    g["poison_lines"] = sorted({f"[POISON] {a} arm={b} survivors={c}" for a, b, c in surv})
    g["poison_lines_n"] = len(surv)
    spins = re.findall(r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+)", text)
    g["spin_success_max"] = max((int(a) for a, _ in spins), default=None)
    g["spin_fail_max"] = max((int(b) for _, b in spins), default=None)
    g["spin_lines_n"] = len(spins)
    # device stamps, 1 tick = 0.01 us; last emission = final soak epoch
    d = re.findall(r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
                   r"servicedrain=(-?\d+) m2_to_end=(-?\d+)", text)
    if d:
        keys = ("planM6", "M7", "combine", "servicedrain", "m2_to_end")
        g["ts_delta_us"] = {k: int(v) / 100.0 for k, v in zip(keys, d[-1])}
        g["ts_delta_n"] = len(d)
        # keep every emission so per-rank spread is visible
        g["ts_delta_all_M7_us"] = sorted(int(x[1]) / 100.0 for x in d)
        g["ts_delta_all_planM6_us"] = sorted(int(x[0]) / 100.0 for x in d)
        g["ts_delta_all_combine_us"] = sorted(int(x[2]) / 100.0 for x in d)
    s = re.findall(r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)", text)
    if s:
        g["ts_split_us"] = {"plan_M3toM5": int(s[-1][0]) / 100.0,
                            "M6": int(s[-1][1]) / 100.0}
        g["ts_split_all_M6_us"] = sorted(int(x[1]) / 100.0 for x in s)
        g["ts_split_n"] = len(s)
    g["log"] = os.path.basename(path)
    return g


def mps_config_from_ranks(outdir):
    """Authoritative per-rank record of what the kernel was actually told.

    K0_MPS_CFG is not echoed anywhere in the campaign summary, and a partial
    config LOOKS like a pass, so the four knobs are read back out of every rank
    JSON of every rotation and required to be identical.
    """
    seen = {}
    soaks = set()
    nfiles = 0
    for rj in sorted(glob.glob(os.path.join(outdir, "run*", "k0pf_mok_synthetic_rank*.json"))):
        try:
            with open(rj) as fh:
                cfg = (json.load(fh) or {}).get("config", {})
        except (OSError, ValueError):
            continue
        mc = cfg.get("mps_config")
        if mc is None:
            continue
        nfiles += 1
        key = json.dumps(mc, sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if "mps_soak_iters" in cfg:
            soaks.add(cfg["mps_soak_iters"])
    out = {"n_rank_files": nfiles,
           "distinct": [json.loads(k) for k in sorted(seen)],
           "counts": sorted(seen.values(), reverse=True),
           "soak_iters": sorted(soaks),
           "consistent": len(seen) == 1}
    if len(seen) == 1:
        out["mps_config"] = json.loads(next(iter(seen)))
    return out


def main():
    tags = sys.argv[1:] or ["e37a", "e37b", "e37c"]
    records = []
    for tag in tags:
        for sj in sorted(glob.glob(os.path.join(HOME, f"k0-mok-{tag}", "*", "summary.json"))):
            outdir = os.path.dirname(sj)
            base = os.path.basename(outdir)
            idx = base.split("_")[0]
            logs = glob.glob(os.path.join(SCRATCH, f"{tag}_{base}.log"))
            with open(sj) as fh:
                summary = json.load(fh)
            rec = {
                "tag": tag,
                "idx": int(idx) if idx.isdigit() else idx,
                "outdir": base,
                "status": summary.get("status"),
                "run_count": summary.get("run_count"),
            }
            arms = summary.get("arm_p50_us", {})
            rec["arm_p50_median"] = {a: v.get("median") for a, v in arms.items()}
            rec["arm_p50_stats"] = {a: v for a, v in arms.items()}
            per_run = []
            for run in summary.get("runs", []):
                per_run.append({
                    "directory": os.path.basename(run.get("directory", "")),
                    "arm_order": run.get("arm_order"),
                    "arm_p50_us": run.get("arm_p50_us"),
                    "arm_p95_us": run.get("arm_p95_us"),
                    "mps_cfg": (run.get("environment") or {}).get("K0_MPS_CFG"),
                    "warmup_iters": run.get("warmup_iters"),
                    "timed_iters": run.get("timed_iters"),
                })
            rec["runs"] = per_run
            cfgs = sorted({r["mps_cfg"] for r in per_run if r["mps_cfg"]})
            rec["mps_cfg"] = cfgs[0] if len(cfgs) == 1 else cfgs
            rec["kernel_hsaco_sha256"] = (summary.get("runs") or [{}])[0].get(
                "kernel_hsaco_sha256")
            rec["cfg_readback"] = mps_config_from_ranks(outdir)
            rec["gates"] = gates_from_log(logs[0]) if logs else {"log": None}
            records.append(rec)
    payload = {"host": os.uname().nodename, "n_campaigns": len(records),
               "campaigns": records}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"wrote {OUT}  campaigns={len(records)}")
    for r in records:
        mc = r["cfg_readback"].get("mps_config", r["cfg_readback"]["distinct"])
        m = r["arm_p50_median"].get("mps_mega")
        p = r["arm_p50_median"].get("production")
        print(f"{r['tag']} idx={r['idx']:>2} C={mc.get('C') if isinstance(mc, dict) else mc}"
              f" g={mc.get('g') if isinstance(mc, dict) else '?'}"
              f" mode={mc.get('mode') if isinstance(mc, dict) else '?'}"
              f" fr={mc.get('flush_rows') if isinstance(mc, dict) else '?'}"
              f" ts={mc.get('timestamps') if isinstance(mc, dict) else '?'}"
              f" soak={r['cfg_readback']['soak_iters']}"
              f" | mps={m and round(m, 1)} prod={p and round(p, 1)}"
              f" ratio={(m / p) if (m and p) else None:.4f}"
              f" | {r['status']} gate={r['gates'].get('mok_gate_pass')}"
              f" ctl={r['gates'].get('control_fails')}"
              f" soakpass={r['gates'].get('soak_pass')}({r['gates'].get('soak_epochs')})"
              f" pperr={r['gates'].get('pperr_max')}"
              f" self={r['gates'].get('poison_selftest_all_true')}"
              f" surv={r['gates'].get('poison_survivors_max')}"
              f" spin={r['gates'].get('spin_success_max')}/{r['gates'].get('spin_fail_max')}"
              f" | M6={r['gates'].get('ts_split_us', {}).get('M6')}"
              f" M7={r['gates'].get('ts_delta_us', {}).get('M7')}"
              f" cmb={r['gates'].get('ts_delta_us', {}).get('combine')}")


if __name__ == "__main__":
    main()
