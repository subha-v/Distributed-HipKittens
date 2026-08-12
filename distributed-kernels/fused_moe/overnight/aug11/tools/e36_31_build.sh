#!/usr/bin/env bash
# exp_36: refresh collection + build the deliverable JSON on the node.
set -u
python3 "$HOME/e36/collect.py" > "$HOME/e36/collected.json" 2> "$HOME/e36/collect.err" || {
  echo "collector failed:"; cat "$HOME/e36/collect.err"; exit 1; }
python3 "$HOME/e36/build_grid.py"
python3 - "$HOME/e36/sensitivity_grid.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = [p for p in d["points"] if p["status"] == "ok"]
print(f"points={len(d['points'])} ok={len(ok)} "
      f"campaigns={sum(p['n_campaigns'] for p in d['points'])}")
for p in sorted(ok, key=lambda p: (p["measurement"], p["ratio_vs_production"])):
    print(f"  {p['measurement']:<9} T={p['T']} seed={p['route_seed_base']} "
          f"{p['config']:<44} {p['p50_us']:8.1f} r={p['ratio_vs_production']:.4f} "
          f"spin={p['spin_success_max']}/{p['spin_fail_max']}")
PY
exit 0
