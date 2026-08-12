#!/usr/bin/env bash
# exp_36: final refresh + the campaign-only table (with phase stamps) that
# result.md quotes.
set -u
python3 "$HOME/e36/collect.py" > "$HOME/e36/collected.json" 2> "$HOME/e36/collect.err" || {
  echo "collector failed:"; cat "$HOME/e36/collect.err"; exit 1; }
python3 "$HOME/e36/build_grid.py"
python3 - "$HOME/e36/sensitivity_grid.json" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
camp = [p for p in d["points"] if p["status"] == "ok" and p["measurement"] == "campaign"]
print("\n=== CAMPAIGN ROWS (5 processes, 500 warmup / 100 timed) ===")
h = (f"{'C':>4}{'g':>5}{'mode':>5}{'depth':>6}{'seed':>6}{'mps_us':>9}{'prod_us':>9}"
     f"{'pf6_us':>9}{'ratio':>8}{'spin':>7}{'M6':>8}{'M7':>8}{'comb':>7}{'plan':>7}")
print(h); print("-" * len(h))
for p in sorted(camp, key=lambda p: (p["route_seed_base"], p["ratio_vs_production"])):
    st = p["phase_stamps"] or {}
    f = lambda v, n=1: ("" if v is None else f"{v:.{n}f}")
    print(f"{p['C']:>4}{p['g']:>5}{p['mode']:>5}{str(p['throttle_depth']):>6}"
          f"{p['route_seed_base']:>6}{p['p50_us']:>9.1f}"
          f"{p['production_p50_us_same_run']:>9.1f}{p['pf6gm_p50_us_same_run']:>9.1f}"
          f"{p['ratio_vs_production']:>8.4f}"
          f"{str(p['spin_success_max'])+'/'+str(p['spin_fail_max']):>7}"
          f"{f(st.get('M6')):>8}{f(st.get('M7')):>8}{f(st.get('combine')):>7}"
          f"{f(st.get('plan')):>7}")

print("\n=== replicate groups (seed 1234) ===")
groups = {}
for p in camp:
    if p["route_seed_base"] != 1234:
        continue
    groups.setdefault((p["C"], p["g"], p["mode"]), []).append(p)
for k in sorted(groups, key=lambda k: statistics.mean(x["p50_us"] for x in groups[k])):
    v = groups[k]
    us = [x["p50_us"] for x in v]
    r = [x["ratio_vs_production"] for x in v]
    print(f"C={k[0]:<3} g={k[1]:<4} mode={k[2]:<3} n={len(us)} "
          f"p50 mean={statistics.mean(us):8.1f} "
          f"vals={[round(x,1) for x in us]} "
          f"ratio mean={statistics.mean(r):.4f} vals={[round(x,4) for x in r]}")

print("\n=== measured route imbalance by seed ===")
seen = {}
for p in d["points"]:
    if p.get("route_std_measured"):
        seen.setdefault(p["route_seed_base"], set()).add(round(p["route_std_measured"], 5))
for s in sorted(seen):
    print(f"  seed_base={s}: dest_load_cv={sorted(seen[s])}")
PY
exit 0
