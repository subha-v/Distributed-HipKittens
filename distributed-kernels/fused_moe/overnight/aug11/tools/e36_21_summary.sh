#!/usr/bin/env bash
# exp_36: refresh the collection and print the compact table.
set -u
python3 "$HOME/e36/collect.py" > "$HOME/e36/collected.json" 2> "$HOME/e36/collect.err" || {
  echo "collector failed:"; cat "$HOME/e36/collect.err"; exit 1; }
python3 - "$HOME/e36/collected.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
rows.sort(key=lambda r: (r["tag"], r["outdir"]))
hdr = f"{'tag':<11}{'T':>6} {'cfg':<40}{'meas':<9}{'green':<6}{'mps_us':>9}{'prod_us':>9}{'pf6_us':>9}{'ratio':>8}{'spin':>8} {'M6':>7}{'M7':>7}{'comb':>7}{'cv':>8}"
print(hdr); print("-" * len(hdr))
for r in rows:
    sh = r["shape"] or {}
    st = r["phase_stamps_us"] or {}
    rt = r["route"] or {}
    f = lambda v, n=1: ("" if v is None else f"{v:.{n}f}")
    print(f"{r['tag']:<11}{str(sh.get('T','?')):>6} {r['cfg']:<40}"
          f"{r['measurement']:<9}{str(r['gates_green']):<6}"
          f"{f(r['p50_us']):>9}{f(r['production_p50_us_same_run']):>9}"
          f"{f(r['pf6gm_p50_us_same_run']):>9}"
          f"{f(r['ratio_vs_production'],4):>8}"
          f"{str(r['spin_success_max'])+'/'+str(r['spin_fail_max']):>8} "
          f"{f(st.get('M6')):>7}{f(st.get('M7')):>7}{f(st.get('combine')):>7}"
          f"{f(rt.get('dest_load_cv'),5):>8}")
PY
exit 0
