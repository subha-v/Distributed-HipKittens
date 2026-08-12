#!/usr/bin/env bash
# exp_36: the same seed should give the same realised route; three different
# dest_load_cv values at seed 1234 need explaining before they go in the JSON.
set -u
python3 - "$HOME/e36/collected.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    rt = r.get("route") or {}
    if not rt:
        continue
    print(f"{r['tag']:<10} {r['measurement']:<9} {r['cfg'][:30]:<32} "
          f"cv={rt.get('dest_load_cv'):.5f} tokens={rt.get('tokens_per_rank')} "
          f"dest={rt.get('dest_load')}")
PY
exit 0
