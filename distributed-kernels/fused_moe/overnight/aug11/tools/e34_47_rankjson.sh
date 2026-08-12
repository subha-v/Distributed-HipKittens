#!/usr/bin/env bash
# exp_34: per-RANK evidence for both ladder runs. The rank-0 stdout cannot show
# a rank-7-only failure, so the negative control's bit-25 claim has to be read
# out of the eight rank JSONs.
set -uo pipefail
for TAG in e34smoke e34neg; do
  D=$(ls -dt $HOME/k0-mok-$TAG/*/ 2>/dev/null | head -1)
  echo "################ $TAG : $D ################"
  ls "$D" | head -20
  python3 - "$D" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
print("json files:", [os.path.basename(f) for f in files][:14])
for f in files:
    if os.path.basename(f) == "summary.json":
        continue
    try:
        j = json.load(open(f))
    except Exception as e:
        print(" ", os.path.basename(f), "unreadable", e); continue
    rank = j.get("rank", j.get("local_rank", "?"))
    out = []
    for arm, v in (j.get("arms") or {}).items():
        g = (v or {}).get("output_gate") or {}
        out.append("%s: pperr=%s pass=%s all_ranks=%s rel=%s" % (
            arm, g.get("pperr"), g.get("pass"), g.get("pass_all_ranks"),
            (str(g.get("relative"))[:8] if g.get("relative") is not None else None)))
    print(f"  rank {rank}: " + " | ".join(out) if out else f"  rank {rank}: keys={list(j)[:12]}")
PY
  echo
done
echo "===DONE==="
