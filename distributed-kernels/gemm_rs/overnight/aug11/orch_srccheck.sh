#!/usr/bin/env bash
# URGENT read-only check: did an unauthorized kernel-source edit reach the node,
# and could it have contaminated exp_23's rung binaries (whose sweep is live)?
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRC=/home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp

echo "===== 1. node kernel source: does it carry the exp_26 edit? ====="
md5sum "$SRC"
echo "-- PERSHAPE macro present? --"
grep -n 'RELEASE_GROUP_PERSHAPE' "$SRC" 2>&1 | head -8
echo "-- source mtime --"
stat -c '%y  %n' "$SRC" 2>&1

echo
echo "===== 2. exp_23 rung binaries: built BEFORE or AFTER the source mtime? ====="
ls -la --time-style=full-iso "$ON"/aug11/exp_23_waterfall/build/*.so 2>&1

echo
echo "===== 3. is the sweep running, and is it rebuilding? ====="
ps -eo pid,etime,stat,cmd 2>/dev/null | grep -E 'sweep\.py|go_sweep|build_rungs|hipcc' | grep -v grep || echo "(no sweep/build process)"

echo
echo "===== 4. GPU lease ====="
bash "$ON/tools/gpu_lease.sh" status

echo
echo "===== 5. fingerprints recorded vs binaries on disk NOW ====="
if [ -f "$ON/aug11/exp_23_waterfall/fingerprints.json" ]; then
python3 - "$ON" <<'PY' 2>&1
import json, hashlib, os, sys
ON = sys.argv[1]
d = json.load(open(f"{ON}/aug11/exp_23_waterfall/fingerprints.json"))
def walk(o, out):
    if isinstance(o, dict):
        if "so_sha256" in o or "so_sha" in o:
            out.append(o)
        for v in o.values(): walk(v, out)
    elif isinstance(o, list):
        for v in o: walk(v, out)
out = []
walk(d, out)
print(f"  fingerprint records found: {len(out)}")
for r in out:
    name = r.get("rung") or r.get("name") or "?"
    rec  = r.get("so_sha256") or r.get("so_sha") or ""
    path = r.get("so") or r.get("so_path") or ""
    if path and os.path.exists(path):
        cur = hashlib.sha256(open(path,'rb').read()).hexdigest()
        ok = "MATCH " if cur.startswith(rec[:12]) else "*** MISMATCH ***"
        print(f"  {name:6} recorded={rec[:12]} on-disk={cur[:12]} {ok}")
    else:
        print(f"  {name:6} recorded={rec[:12]} (path not resolvable: {path!r})")
PY
else
  echo "  fingerprints.json absent"
fi
echo "done"
