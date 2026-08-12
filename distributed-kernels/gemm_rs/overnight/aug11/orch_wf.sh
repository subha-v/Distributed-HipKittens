#!/usr/bin/env bash
# Read-only: pull exp_23's per-shape waterfall numbers and the fingerprint
# assertion state that was in force WHEN THE CAMPAIGN RAN.
set -uo pipefail
A=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_23_waterfall

echo "===== fingerprints.json: assertion state (current) ====="
python3 - "$A" <<'PY' 2>&1
import json, sys
A = sys.argv[1]
d = json.load(open(f"{A}/fingerprints.json"))
def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if key in k.lower(): out.append((k, v))
            find(v, key, out)
    elif isinstance(o, list):
        for v in o: find(v, key, out)
out = []
find(d, "assert", out)
for k, v in out[:6]:
    print(f"  {k}: {json.dumps(v)[:600]}")
print("  top-level keys:", list(d.keys()))
for k in ("generated_utc","all_pass","hard_pass","exit"):
    if k in d: print(f"  {k} = {d[k]}")
PY

echo
echo "===== stats.json: per-shape contrasts ====="
python3 - "$A" <<'PY' 2>&1
import json, sys
A = sys.argv[1]
try:
    s = json.load(open(f"{A}/stats.json"))
except Exception as e:
    print("  no stats.json:", e); raise SystemExit
print("  keys:", list(s.keys()))
print(json.dumps(s, indent=1)[:4000])
PY
echo done
