#!/usr/bin/env bash
# Inspect the official evaluator setup that exp026 already ran successfully on
# this node, so the frozen rank-1 submission can be benchmarked with zero new
# engineering.
set -uo pipefail

EXP=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29
CWD=$EXP/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
SPEC=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/inputs/specs/reference-kernels

echo "===== proven evaluator cwd ====="
ls -la "$CWD" 2>&1

echo
echo "===== cases.txt (test) ====="
cat "$CWD/cases.txt" 2>&1 | head -20

echo
echo "===== environment used by exp026 ====="
python3 - <<'PY'
import json
p = ("/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/"
     "exp026-stock-gemm-rs-test-de730f29/index.json")
d = json.load(open(p))
ex = d.get("execution", {})
print("argv:", ex.get("argv"))
print("cwd:", ex.get("cwd"))
env = ex.get("environment", {})
for k in sorted(env):
    print(f"  {k} = {env[k]}")
PY

echo
echo "===== spec source of truth ====="
ls -la "$SPEC" 2>&1 | head -20
echo "-- gemm-rs --"
ls -la "$SPEC/gemm-rs" 2>&1

echo
echo "===== eval.py: how modes are selected ====="
grep -nE 'def main|sys.argv|benchmark|"test"|leaderboard|ranked|mode' "$CWD/eval.py" 2>&1 | head -40

echo
echo "===== eval.py: how many lines, and does it read a cases file ====="
wc -l "$CWD"/*.py 2>&1

echo
echo "===== frozen rank-1 submission ====="
ls -la /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/
sha256sum /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
cat /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/SHA256SUMS.txt

echo
echo "===== does the current submission.py differ from rank-1? ====="
sha256sum "$CWD/submission.py" 2>&1
head -20 "$CWD/submission.py" 2>&1
