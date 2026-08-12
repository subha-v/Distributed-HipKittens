#!/usr/bin/env bash
# Shape 3 landed 8/8 rank files but `parse` returned rc=1, so the six-shape
# aggregate is refusing. Get the exact assertion -- this is the parser doing its
# job, so read it rather than reaching for --lenient.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders

echo "===== the aggregation failure, verbatim ====="
sed -n '/=== aggregate/,$p' "$D/logs/full_run.log" | head -60

echo
echo "===== where does the OLD drain string still live? ====="
grep -n 'waiting for kfd drain' "$D/run_ladders.sh"

echo
echo "===== re-run the aggregation by hand to see the traceback ====="
docker exec dhk-gemmrs bash -lc "cd $D && python3 ladders.py --root $D --out /tmp/lad_probe.json --ladder-dir ladder --expect-a 6 --expect-b 0" 2>&1 | tail -40

echo
echo "===== what do shape 3's rank files actually contain? ====="
docker exec dhk-gemmrs python3 - <<'PY'
import glob, json
D="/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders/raw/ladder"
for i in (3, 0):
    fs = sorted(glob.glob(f"{D}/lad_s{i}.rank*.json"))
    print(f"--- shape idx {i}: {len(fs)} files")
    if not fs: continue
    d = json.load(open(fs[0]))
    print("   top keys :", sorted(d.keys()))
    print("   shape_idx:", d.get("shape_index"), " spec:", d.get("shape"))
    for a in sorted(d.get("arms", {})):
        for p in sorted(d["arms"][a]):
            n = len(d["arms"][a][p].get("samples_us", []))
            print(f"   {a:<14} {p:<10} n={n}")
        break
    print("   arms:", sorted(d.get("arms", {})))
    print("   correctness:", d.get("correctness"))
PY
