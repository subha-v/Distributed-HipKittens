#!/usr/bin/env python3
"""exp_38: collect the GPU confirmation into a clean table + plot-ready JSON.

Two required checks, and the second is the one that proves the fix rather than a
timing coincidence:
  (1) the end-to-end ratchet reads ~6,482.7 us / ~0.8408x production
  (2) the exp_24 injection bound is live again: g=353 minus g=65 is back near
      -613 us, not the -1.8 us the mode-14 pin showed.
Medians, not means, and the paired contrast is computed on the medians of the
same-session alternating rounds.
"""
import csv, json, os, statistics as st

CSV = os.path.expanduser("~/overnight-scratch/screen_e38a.csv")
DST = os.path.expanduser("~/overnight-scratch/e38/ratchet_confirm.json")

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if r["status"] != "OK":
            print(f"  !! idx {r['idx']} status={r['status']} -- excluded")
            continue
        g = int(r["cfg"].split("g=")[1].split(",")[0])
        rows.append({
            "idx": int(r["idx"]), "utc": r["utc"], "cfg": r["cfg"], "g": g,
            "head": r["head"], "src_rev": int(r["src_rev"]),
            "prod_us": float(r["prod_us"]), "pf6gm_us": float(r["pf6gm_us"]),
            "mps_us": float(r["mps_us"]),
            "ratio_vs_prod": float(r["ratio_vs_prod"]),
            "gate_pass": r["gate_pass"], "control_fails": r["control_fails"],
            "pperr": int(r["pperr"]), "soak": r["soak_epochs"],
            "mps_hsaco": r["mps_hsaco"],
        })
rows.sort(key=lambda x: x["idx"])

print("=== exp_38 GPU confirmation (STAMPS OFF, 5 proc x 500 warmup / 100 timed) ===")
print("  %-4s %-6s %-9s %-9s %-9s %-8s %-6s %-6s %-6s %s"
      % ("idx", "g", "prod_us", "pf6gm_us", "mps_us", "vs_prod",
         "gate", "ctl", "pperr", "soak"))
for r in rows:
    print("  %-4d %-6d %-9.1f %-9.1f %-9.1f %-8.4f %-6s %-6s %-6d %s"
          % (r["idx"], r["g"], r["prod_us"], r["pf6gm_us"], r["mps_us"],
             r["ratio_vs_prod"], r["gate_pass"], r["control_fails"],
             r["pperr"], r["soak"]))

by_g = {}
for r in rows:
    by_g.setdefault(r["g"], []).append(r)

print("\n=== medians by arm ===")
summ = {}
for g in sorted(by_g, reverse=True):
    v = by_g[g]
    m = st.median([x["mps_us"] for x in v])
    summ[g] = {
        "n": len(v),
        "mps_us_median": round(m, 1),
        "mps_us_all": [x["mps_us"] for x in v],
        "prod_us_median": round(st.median([x["prod_us"] for x in v]), 1),
        "pf6gm_us_median": round(st.median([x["pf6gm_us"] for x in v]), 1),
        "ratio_vs_prod_median": round(st.median([x["ratio_vs_prod"] for x in v]), 4),
        "spread_us": round(max(x["mps_us"] for x in v) - min(x["mps_us"] for x in v), 1),
    }
    s = summ[g]
    label = "throttle ON  (ratchet)" if g == 353 else "throttle OFF"
    print(f"  g={g:<4} {label:<24} n={s['n']}  mps={s['mps_us_median']} us  "
          f"ratio={s['ratio_vs_prod_median']}  prod={s['prod_us_median']}  "
          f"spread={s['spread_us']} us")

print("\n=== CHECK 1: the ratchet ===")
r353 = summ[353]["mps_us_median"]
q353 = summ[353]["ratio_vs_prod_median"]
print(f"  measured : {r353} us  =  {q353}x production")
print(f"  expected : ~6482.7 us =  ~0.8408x   (the mode-14 pin read 7224.2 / 0.938)")
print(f"  delta vs expected: {r353 - 6482.7:+.1f} us")
print("  VERDICT: RESTORED" if abs(r353 - 6482.7) < 30 else "  VERDICT: OFF TARGET")

print("\n=== CHECK 2: the exp_24 injection bound (the mechanistic one) ===")
contrast = r353 - summ[65]["mps_us_median"]
print(f"  g=353 - g=65 = {r353} - {summ[65]['mps_us_median']} = {contrast:+.1f} us")
print(f"  rev 26 read -613.5 us; the mode-14 pin read -1.8 us (INERT)")
print("  VERDICT: THROTTLE LIVE AGAIN" if contrast < -400 else "  VERDICT: STILL INERT")

out = {
    "experiment": "exp_38_ratchet_restore",
    "arm": "default build of codex/distributed-hipkittens-scaffold",
    "head": rows[0]["head"], "src_rev": rows[0]["src_rev"],
    "timestamps": "off",
    "iters": "5 proc x 500 warmup / 100 timed, K0_MPS_SOAK_ITERS=600",
    "runs": rows,
    "by_arm": {str(k): v for k, v in summ.items()},
    "check_1_ratchet": {
        "measured_us": r353, "measured_ratio_vs_prod": q353,
        "expected_us": 6482.7, "expected_ratio": 0.8408,
        "mode14_pin_us": 7224.2,
        "pass": abs(r353 - 6482.7) < 30,
    },
    "check_2_injection_bound": {
        "contrast_us": round(contrast, 1),
        "rev26_us": -613.5, "mode14_pin_us": -1.8,
        "pass": contrast < -400,
    },
}
os.makedirs(os.path.dirname(DST), exist_ok=True)
json.dump(out, open(DST, "w"), indent=2)
print(f"\nwrote {DST}")
