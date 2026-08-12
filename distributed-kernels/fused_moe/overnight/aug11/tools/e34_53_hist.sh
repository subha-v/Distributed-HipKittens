#!/usr/bin/env bash
# exp_34 sanity: the same-session mode-12 control reads ~7,222 us where the
# published aug11 ratchet is 6,568.0. Pull EVERY historical SCREEN csv row for a
# mode-12 C=16 g=353 config off this node, with its commit and hsaco hash, to see
# whether 7,222 is new with the mode-14 commit or older than it.
set -uo pipefail
echo "===tree cleanliness at the pin==="
git -C "$HOME/Distributed-HipKittens" rev-parse HEAD
git -C "$HOME/Distributed-HipKittens" status --porcelain | head -20
echo "(empty porcelain above = clean)"
echo
echo "===all screen csv rows mentioning mode=12 (any tag)==="
python3 - <<'PY'
import glob, os, re
HOME = os.path.expanduser("~")
rows = []
for f in glob.glob(f"{HOME}/overnight-scratch/*.csv") + glob.glob(f"{HOME}/overnight-scratch/*.log") \
        + glob.glob(f"{HOME}/e3*/*.csv") + glob.glob(f"{HOME}/*.csv"):
    try:
        t = open(f, errors="ignore").read()
    except Exception:
        continue
    for line in t.splitlines():
        if line.startswith("SCREEN ") and "mode=12" in line and "g=353" in line:
            rows.append((os.path.basename(f), line))
seen = set()
for f, line in rows:
    p = line.split(",")
    key = tuple(p[1:9])
    if key in seen:
        continue
    seen.add(key)
    # idx,ts,cfg(quoted -> may split),status...  just print the interesting slice
    print(f"{f:34s} {line[:230]}")
print(f"\n{len(seen)} distinct mode=12 g=353 rows")
PY
echo "===DONE==="
