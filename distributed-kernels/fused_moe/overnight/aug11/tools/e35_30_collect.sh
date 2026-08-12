#!/usr/bin/env bash
# exp_35 step 7: collect one JSON record per campaign -- full arm_p50_us
# (median + per-process values for EVERY arm, so every rung has a same-run
# paired denominator), the literal gate lines, and the device phase stamps.
# Usage: bash e35_30_collect.sh TAG [TAG2 ...]
set -uo pipefail
OUTJ="$HOME/e35/collected.json"
python3 - "$@" <<'PY' | tee "$OUTJ" | tail -5
import glob, json, os, re, statistics, sys

HOME = os.path.expanduser("~")
tags = sys.argv[1:] or ["e35w"]
records = []

def lastn(pat, text, n=None):
    return re.findall(pat, text)

for tag in tags:
    for log in sorted(glob.glob(f"{HOME}/overnight-scratch/{tag}_*.log")):
        base = os.path.basename(log)[:-4]                    # TAG_IDX_SLUG_STAMP
        suffix = base[len(tag) + 1:]                         # IDX_SLUG_STAMP
        outdir = f"{HOME}/k0-mok-{tag}/{suffix}"
        text = open(log, errors="ignore").read()

        rec = {"tag": tag, "run_id": suffix, "log": log, "output_root": outdir}

        # ---- the config actually requested -----------------------------------
        m = re.search(r"K0_MPS_CFG=(\S+)", text)
        rec["k0_mps_cfg"] = m.group(1) if m else None

        # ---- timings: median over processes of rank-max p50, plus the raw -----
        sj = os.path.join(outdir, "summary.json")
        rec["summary_json"] = sj if os.path.exists(sj) else None
        rec["arm_p50_us"] = {}
        if rec["summary_json"]:
            s = json.load(open(sj))
            for arm, st in s.get("arm_p50_us", {}).items():
                rec["arm_p50_us"][arm] = {
                    "median": st.get("median"),
                    "values": st.get("values"),
                    "n_processes": len(st.get("values") or []),
                }
            rec["runs_in_campaign"] = len(s.get("runs", []) or [])

        # ---- literal gate lines (quoted verbatim in result.md) ---------------
        def lines(pat):
            return sorted(set(l.strip() for l in re.findall(pat, text, re.M)))
        rec["gates"] = {
            "mok_gate":  lines(r"^.*\[MOK GATE\] mps_mega .*$"),
            "mark":      lines(r"^.*\[MARK\] control_fails=\w+.*$"),
            "poison_selftest": lines(r"^.*\[POISON SELFTEST\].*$"),
            "poison":    lines(r"^.*\[POISON\] .*survivors=\d+.*$"),
            "soak":      lines(r"^.*\[MPS SOAK\] .*$"),
            "eager":     lines(r"^.*\[MOK SYNTHETIC EAGER\] .*$"),
        }
        perrs = [int(v) for v in re.findall(r"pperr=(\d+)", text)]
        rec["pperr_max"] = max(perrs) if perrs else None
        rec["pperr_distinct"] = sorted(set(perrs))
        surv = [int(v) for v in re.findall(r"\[POISON\] \S+ arm=\S+ rank=\d+ survivors=(\d+)", text)]
        rec["poison_survivors_max"] = max(surv) if surv else None
        st_bad = [a for a, v in re.findall(
            r"\[POISON SELFTEST\] arm=(\S+) one_row_poisoned_fails=(\w+)", text) if v != "True"]
        rec["poison_selftest_dead"] = st_bad
        soak_ok = re.findall(r"\[MPS SOAK\] completed=(\d+)/(\d+) pperr=(\d+).*?pass=(\w+)", text)
        rec["soak_epochs"] = sorted(set(f"{a}/{b}" for a, b, c, d in soak_ok))
        rec["soak_pass_all"] = bool(soak_ok) and all(d == "True" for *_, d in soak_ok)
        gate_ok = re.findall(r"\[MOK GATE\] mps_mega max_abs=\S+ relative=\S+ pass=(\w+)", text)
        rec["mok_gate_pass_all"] = bool(gate_ok) and all(v == "True" for v in gate_ok)
        ctl = re.findall(r"\[MARK\] control_fails=(\w+)", text)
        rec["control_fails_all"] = bool(ctl) and all(v == "True" for v in ctl)
        rec["gates_green"] = bool(
            rec["mok_gate_pass_all"] and rec["control_fails_all"] and rec["soak_pass_all"]
            and rec["pperr_max"] == 0 and not rec["poison_selftest_dead"]
            and (rec["poison_survivors_max"] in (0, None)))

        # ---- device phase stamps. 1 tick = 0.01 us; these are MAX stamps from
        # ---- the final soak epoch, so they do NOT sum to arm_p50_us.
        def stamp(pat, keys):
            hits = re.findall(pat, text)
            if not hits:
                return None
            cols = list(zip(*[[int(x) for x in h] for h in hits]))
            return {k: {"median_us": round(statistics.median(c) / 100.0, 1),
                        "values_us": [round(v / 100.0, 1) for v in c],
                        "n": len(c)}
                    for k, c in zip(keys, cols)}
        rec["ts_delta"] = stamp(
            r"\[MPS TS DELTA\] planM6=(-?\d+) M7=(-?\d+) combine=(-?\d+) "
            r"servicedrain=(-?\d+) m2_to_end=(-?\d+)",
            ["planM6", "M7", "combine", "servicedrain", "m2_to_end"])
        rec["ts_split"] = stamp(
            r"\[MPS TS SPLIT\] plan_M3toM5=(-?\d+) M6=(-?\d+)",
            ["plan_M3toM5", "M6"])
        rec["spin"] = stamp(
            r"\[MPS SPIN\] chunk_poll success_max=(\d+) fail_max=(\d+)",
            ["success_max", "fail_max"])
        rec["ts_raw_lines"] = sorted(set(l.strip() for l in re.findall(
            r"^.*\[MPS TS(?: SPLIT| DELTA)?\] .*$|^.*\[MPS SPIN\] .*$", text, re.M)))[:12]

        # ---- descriptor dump: the packed cfg word the KERNEL actually read ----
        rec["desc_dump"] = {}
        for dd in sorted(glob.glob(os.path.join(outdir, "**", "mps_desc_rank*.txt"),
                                   recursive=True)):
            try:
                vals = json.loads(open(dd).read().replace("'", '"'))
            except Exception:
                continue
            rec["desc_dump"][os.path.basename(dd)] = {"n_slots": len(vals), "slots": vals}
        records.append(rec)

print(json.dumps({"records": records}, indent=1))
PY
echo
echo "===WROTE $OUTJ ($(wc -c < "$OUTJ") bytes, $(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["records"]))' "$OUTJ") records)==="
