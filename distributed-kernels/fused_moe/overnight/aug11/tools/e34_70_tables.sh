#!/usr/bin/env bash
# exp_34: render result.md's four tables straight out of mode14_arms.json, so no
# number in the write-up is ever hand-typed.
set -uo pipefail
python3 - <<'PY'
import json, os

d = json.load(open(os.path.expanduser("~/e34/mode14_arms.json")))
arms = {r["arm_id"]: r for r in d["arms"]}

def g(aid):
    return arms.get(aid)

def num(v, s="%.1f"):
    return (s % v) if isinstance(v, (int, float)) else "—"

def com(v):
    return f"{v:,.1f}" if isinstance(v, (int, float)) else "—"

def sgn(v):
    return (f"{v:+,.1f}" if isinstance(v, (int, float)) else "—")

LABELS_OFF = [
    ("rev26_mode12_C16_g353_ts0", "**true ratchet (in-session)**"),
    ("rev28_mode12_C16_g353_ts0", "mode-12 control **at the pin**"),
    ("rev28_mode12_C16_g65_ts0",  "mode 12 unthrottled (`g=65`)"),
    ("rev28_mode14_C0_g353_ts0",  "**mode 14, granularity + drain deletion**"),
    ("rev28_mode14_C0_g481_ts0",  "**mode 14, granularity alone (drain kept)**"),
    ("rev28_mode14_C0_g65_ts0",   "mode 14 unthrottled (`g=65`)"),
    ("rev28_mode14_C8_g353_ts0",  "mode 14, C=8"),
    ("rev28_mode14_C16_g353_ts0", "mode 14, C=16"),
]
LABELS_ON = [
    ("rev26_mode12_C16_g353_ts1", "true ratchet (in-session)"),
    ("rev28_mode12_C16_g353_ts1", "mode-12 control at the pin"),
    ("rev28_mode12_C16_g65_ts1",  "mode 12 unthrottled"),
    ("rev28_mode14_C0_g353_ts1",  "mode 14, drain deleted"),
    ("rev28_mode14_C0_g481_ts1",  "mode 14, drain kept"),
    ("rev28_mode14_C0_g65_ts1",   "mode 14 unthrottled"),
    ("rev28_mode14_C8_g353_ts1",  "mode 14, C=8"),
    ("rev28_mode14_C16_g353_ts1", "mode 14, C=16"),
]

print("<<<TABLE-STAMPS-OFF>>>")
print("| arm | rev | mode | C | g | n | p50 µs | individual campaigns | prod "
      "(same run) | ratio | Δ vs pin mode-12 | Δ vs true ratchet |")
print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
for aid, lab in LABELS_OFF:
    r = g(aid)
    if not r:
        continue
    vals = " / ".join(com(c["mps_mega_p50_us"]) for c in r["campaigns"]
                      if c["gates_green"])
    print(f"| {lab} | {r['src_rev']} | {r['mode']} | {r['C']} | {r['g']} | "
          f"{r['n_campaigns']} | **{com(r['p50_median'])}** | {vals} | "
          f"{com(r['production_p50_same_run'])} | "
          f"{num(r['ratio_vs_production'], '%.4f')} | "
          f"{sgn(r['delta_vs_mode12_control'])} | "
          f"{sgn(r['delta_vs_ratchet_rev26'])} |")

print()
print("<<<TABLE-STAMPS-ON>>>")
print("| arm | rev | mode | C | g | n | p50 µs | planM6 | M7 | combine | servicedrain |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for aid, lab in LABELS_ON:
    r = g(aid)
    if not r:
        continue
    ps = r.get("phase_stamps") or {}
    drain = ("**unwritten**" if r["mode"] == 14 else com(ps.get("servicedrain")))
    print(f"| {lab} | {r['src_rev']} | {r['mode']} | {r['C']} | {r['g']} | "
          f"{r['n_campaigns']} | {com(r['p50_median'])} | "
          f"{com(ps.get('planM6'))} | **{com(ps.get('M7'))}** | "
          f"{com(ps.get('combine'))} | {drain} |")

print()
print("<<<TABLE-RUNGS>>>")
b = g("rev28_mode12_C16_g353_ts0")
c = g("rev28_mode14_C0_g481_ts0")
e = g("rev28_mode14_C0_g353_ts0")
print(f"| rung | config | p50 (stamps-off) | n | rung delta |")
print(f"|---|---|---:|---:|---:|")
print(f"| (b) mode 12 at the pin | `C=16,g=353,mode=12` | {com(b['p50_median'])} | "
      f"{b['n_campaigns']} | — |")
print(f"| (c) **granularity alone**, drain retained | `C=0,g=481,mode=14` | "
      f"{com(c['p50_median'])} | {c['n_campaigns']} | "
      f"**{sgn(c['p50_median'] - b['p50_median'])}** |")
print(f"| (d) **+ drain deletion** | `C=0,g=353,mode=14` | {com(e['p50_median'])} | "
      f"{e['n_campaigns']} | **{sgn(e['p50_median'] - c['p50_median'])}** |")

print()
print("<<<TABLE-CSWEEP>>>")
print("| C | p50 (stamps-off) | n | Δ vs C=0 | per reserved CTA |")
print("|---:|---:|---:|---:|---:|")
base = g("rev28_mode14_C0_g353_ts0")["p50_median"]
for C, aid in ((0, "rev28_mode14_C0_g353_ts0"), (8, "rev28_mode14_C8_g353_ts0"),
               (16, "rev28_mode14_C16_g353_ts0")):
    r = g(aid)
    if not r:
        continue
    dv = r["p50_median"] - base
    print(f"| {C} | {com(r['p50_median'])} | {r['n_campaigns']} | "
          f"{'—' if C == 0 else sgn(dv)} | "
          f"{'—' if C == 0 else '%+.1f µs' % (dv / C)} |")
PY
echo "===DONE==="
