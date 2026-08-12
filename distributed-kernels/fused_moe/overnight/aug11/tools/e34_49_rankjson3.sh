#!/usr/bin/env bash
# exp_34: find EVERY pperr / poison / gate leaf per rank, recursively.
set -uo pipefail
for TAG in e34smoke e34neg; do
  D=$(ls -dt $HOME/k0-mok-$TAG/*/ 2>/dev/null | head -1)
  echo "################ $TAG ################"
  python3 - "$D" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:8]):
            yield from walk(v, path + f"[{i}]")
    else:
        yield path, o
for f in sorted(glob.glob(os.path.join(d, "**", "k0pf_mok_synthetic_rank*.json"), recursive=True)):
    j = json.load(open(f))
    rank = j.get("rank")
    hits = [(p, v) for p, v in walk(j)
            if ("pperr" in p.lower() or "survivor" in p.lower()
                or p.lower().endswith("/pass") or "pass_all_ranks" in p.lower())
            and "mps" in p.lower()]
    print(f"--- rank {rank} ({os.path.basename(f)}) ---")
    for p, v in hits[:24]:
        print("   ", p, "=", v)
PY
  echo
done
echo "===DONE==="
