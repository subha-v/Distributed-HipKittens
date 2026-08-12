#!/usr/bin/env bash
# exp_34: per-rank pperr / gate, take 2 (arms is a LIST in these JSONs).
set -uo pipefail
for TAG in e34smoke e34neg; do
  D=$(ls -dt $HOME/k0-mok-$TAG/*/ 2>/dev/null | head -1)
  echo "################ $TAG ################"
  python3 - "$D" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
f0 = sorted(glob.glob(os.path.join(d, "**", "k0pf_mok_synthetic_rank0.json"), recursive=True))[0]
j = json.load(open(f0))
print("top-level keys:", list(j)[:20])
a = j.get("arms")
print("type(arms) =", type(a).__name__)
if isinstance(a, list) and a:
    print("first arm record keys:", list(a[0])[:25])
    print(json.dumps(a[0], indent=1)[:1200])
PY
  echo
done
echo "===DONE==="
