#!/usr/bin/env bash
# exp_27 BIT-IDENTITY EVIDENCE, at full float precision from the rank JSONs
# rather than the 6 rounded digits the [MOK GATE] line prints.
#
# WHY THE REQUESTED TEST CANNOT BE RUN AS STATED
# The brief asks for the candidate to reproduce the ratchet's printed
# `max_abs=0.035156 relative=0.008293` EXACTLY. It cannot, and neither can the
# ratchet: batch 1 ran a binary whose .text is byte-identical to the ratchet
# (fingerprinted `ab0c353b...`) five times and printed TWO different `relative`
# values, and `production` -- an arm exp_27 does not touch -- printed two
# different max_abs values in the same five runs. The MoK dispatch assigns
# receive rows by fetch_add on a device counter, so arrival order, and therefore
# the top-k reduction order, differs run to run. The inputs are not reproducible,
# so no cross-run digit comparison is a gate.
#
# WHAT IS SOUND, AND IS WHAT THIS SCRIPT MEASURES
# `pf6gm_mega` is an untouched arm evaluated on the SAME input in the SAME run, so
# it is a per-run fingerprint of that run's input and reference. The test is
# therefore the SAME-RUN PAIRED quantity
#
#     d = mps_mega.relative_error - pf6gm_mega.relative_error
#
# over every (run, rank) cell, compared BETWEEN ARMS. If the layout switch changed
# which numbers M6 reads, d shifts. If d has the same distribution in both arms,
# the numerics are unchanged to the resolution this harness can offer -- and the
# script also reports the input-drift floor (the spread of the untouched
# `production` arm) so that resolution is stated rather than assumed.
set -uo pipefail
python3 - "$@" <<'PY'
import json, glob, os, sys, statistics as st

BATCH = {"e27b1_ctl": 0, "e27b2_cand": 1, "e27b3_ctl": 0, "e27b4_cand": 1}
rows = []
for tag, arm in BATCH.items():
    for rj in sorted(glob.glob(os.path.expanduser(f"~/k0-mok-{tag}/*/run*/k0pf_mok_synthetic_rank*.json"))):
        try: d = json.load(open(rj))
        except Exception: continue
        run = os.path.basename(os.path.dirname(os.path.dirname(rj))).split("_")[0]
        rank = int(os.path.basename(rj).split("rank")[1].split(".")[0])
        mc = d.get("mok_correctness", {})
        eg = d.get("eager", {})
        def g(sec, arm_, key):
            return sec.get(arm_, {}).get(key)
        rows.append(dict(
            tag=tag, arm=arm, run=run, rank=rank,
            mps=g(mc, "mps_mega", "relative_error"),
            pf6=g(mc, "pf6gm_mega", "relative_error"),
            prod=g(mc, "production", "relative_error"),
            mps_l2=g(eg, "mps_mega", "rel_L2"), pf6_l2=g(eg, "pf6gm_mega", "rel_L2"),
            prod_l2=g(eg, "production", "rel_L2"),
            mps_abs=g(eg, "mps_mega", "max_abs"), pf6_abs=g(eg, "pf6gm_mega", "max_abs"),
            soak=d.get("mps_soak", {}).get("correctness", {}).get("relative_error"),
            ctl=d.get("mok_control", {}).get("relative_error"),
            nf=g(mc, "mps_mega", "nonfinite"),
        ))

rows = [r for r in rows if r["mps"] is not None and r["pf6"] is not None]
if not rows:
    print("no rank JSONs found yet"); sys.exit(0)

def desc(name, v, sig=4):
    if len(v) < 2: return f"{name:34s} n={len(v)}  {v}"
    return (f"{name:34s} n={len(v):3d}  mean={st.mean(v):+.{sig}e}  "
            f"sd={st.stdev(v):.{sig}e}  min={min(v):+.{sig}e}  max={max(v):+.{sig}e}")

print("=" * 104)
print("1. THE INPUT IS NOT RUN-REPRODUCIBLE -- the resolution floor of any digit comparison")
print("=" * 104)
for arm in (0, 1):
    p = [r["prod"] for r in rows if r["arm"] == arm]
    f = [r["pf6"] for r in rows if r["arm"] == arm]
    print(desc(f"  arm {arm}: production rel_err", p))
    print(desc(f"  arm {arm}: pf6gm_mega rel_err", f))
print("  ^ `production` and `pf6gm_mega` are BOTH untouched by exp_27. Their spread across")
print("    runs is the run-to-run input drift, i.e. the smallest difference any cross-run")
print("    comparison could possibly resolve.")

print()
print("=" * 104)
print("2. THE GATE: same-run paired  d = mps_mega.rel_err - pf6gm_mega.rel_err,  by arm")
print("=" * 104)
D = {}
for arm in (0, 1):
    D[arm] = [r["mps"] - r["pf6"] for r in rows if r["arm"] == arm]
    print(desc(f"  arm {arm} (0=group-major, 1=token-major)", D[arm]))
if len(D[0]) >= 2 and len(D[1]) >= 2:
    m0, m1 = st.mean(D[0]), st.mean(D[1])
    s0, s1 = st.stdev(D[0]), st.stdev(D[1])
    n0, n1 = len(D[0]), len(D[1])
    se = (s0 * s0 / n0 + s1 * s1 / n1) ** 0.5
    print(f"\n  delta of d between arms : {m1 - m0:+.4e}   (Welch se {se:.4e}, "
          f"t = {(m1 - m0) / se if se else float('nan'):+.2f})")
    prod = [r["prod"] for r in rows]
    drift = st.stdev(prod) if len(prod) > 1 else float("nan")
    print(f"  input-drift floor (sd of untouched production rel_err) : {drift:.4e}")
    print(f"  ratio |arm effect| / input drift : {abs(m1 - m0) / drift:.3f}"
          if drift == drift and drift else "")

print()
print("=" * 104)
print("3. max_abs: is mps_mega EXACTLY equal to untouched pf6gm_mega, per run/rank?")
print("=" * 104)
for arm in (0, 1):
    sub = [r for r in rows if r["arm"] == arm and r["mps_abs"] is not None]
    eq = sum(1 for r in sub if r["mps_abs"] == r["pf6_abs"])
    vals = sorted(set(r["mps_abs"] for r in sub))
    print(f"  arm {arm}: mps_abs == pf6_abs in {eq}/{len(sub)} cells;  "
          f"distinct mps max_abs values = {vals}")

print()
print("=" * 104)
print("4. THE DYNAMIC RANGE OF THE INSTRUMENT -- what a real layout error would read")
print("=" * 104)
ctl = [r["ctl"] for r in rows if r["ctl"] is not None]
mps = [r["mps"] for r in rows]
if ctl:
    print(desc("  negative control rel_err", ctl))
print(desc("  mps_mega rel_err (both arms)", mps))
if ctl:
    print(f"  the deliberately-broken control sits {st.mean(ctl)/st.mean(mps):.0f}x above the arm.")
    print("  A wrong scale layout mis-scales whole 128-channel groups per token, which lands")
    print("  in the control's regime, not 1e-9 away. The instrument cannot miss it.")

print()
print("=" * 104)
print("5. PER-BATCH ROLL-UP")
print("=" * 104)
print(f"  {'tag':12s} {'arm':>3s} {'cells':>5s} {'nonfinite':>9s} "
      f"{'mean mps rel_err':>18s} {'mean d = mps-pf6':>18s}")
for tag, arm in BATCH.items():
    sub = [r for r in rows if r["tag"] == tag]
    if not sub: continue
    print(f"  {tag:12s} {arm:3d} {len(sub):5d} {sum(r['nf'] or 0 for r in sub):9d} "
          f"{st.mean([r['mps'] for r in sub]):18.12f} "
          f"{st.mean([r['mps'] - r['pf6'] for r in sub]):+18.4e}")
PY
