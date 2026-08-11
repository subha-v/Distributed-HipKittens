#!/usr/bin/env bash
set -uo pipefail
CWD=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd

echo "===== cases.txt ====="
cat -A "$CWD/cases.txt" | head -15

echo
echo "===== eval.py main() and case parsing ====="
sed -n '1,60p' "$CWD/eval.py"

echo
echo "----- get_test_cases + main -----"
awk '/def get_test_cases/,/^def [a-z_]+\(/{print NR": "$0}' "$CWD/eval.py" | head -45
echo "..."
sed -n '525,578p' "$CWD/eval.py"

echo
echo "===== utils.py: env / world size / popcorn output ====="
grep -nE 'POPCORN|environ|world_size|WORLD|multiprocessing|set_start_method' "$CWD/eval.py" "$CWD/utils.py" | head -40

echo
echo "===== exp026 environment (full) ====="
python3 - <<'PY'
import json
p = ("/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/"
     "exp026-stock-gemm-rs-test-de730f29/index.json")
d = json.load(open(p))
ex = d.get("execution", {})
print("argv:", ex.get("argv"))
for k, v in sorted(ex.get("environment", {}).items()):
    print(f"  {k} = {v}")
PY
