#!/usr/bin/env bash
# Summarise whatever ours-vs-rank1 samples exist, per shape and per arm.
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1

docker exec dhk-gemmrs python3 - "$E/vs_logs" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "vs_s*.rank0.json")))
if not files:
    print("no rank0 json yet"); raise SystemExit(0)

def stats(xs):
    xs = sorted(xs); n = len(xs)
    return xs[0], xs[n//2], sum(xs)/n

print(f"{'shape':>6} {'arm':>10} {'best':>10} {'median':>10} {'mean':>10}  n")
print("-"*56)
for f in files:
    tag = os.path.basename(f).split('.')[0]
    blob = json.load(open(f))
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, list) and len(o) >= 8 and all(isinstance(x,(int,float)) for x in o):
            yield path, [float(x) for x in o]
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, f"{path}[{i}]")
    for key, xs in walk(blob):
        if len(xs) < 8:
            continue
        b, m, mu = stats(xs)
        print(f"{tag:>6} {key[-10:]:>10} {b:10.2f} {m:10.2f} {mu:10.2f}  {len(xs)}")
PY
echo "===== DONE ====="
