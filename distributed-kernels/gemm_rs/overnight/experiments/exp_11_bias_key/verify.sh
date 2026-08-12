#!/usr/bin/env bash
# Does the evaluator force has_bias=True on every graded shape, and if so does
# our shape table fall off its tuned rows onto the generic fallback?
#
# If both are true, the tuned entries for shapes 1, 4 and 6 are DEAD CODE under
# the real evaluator, and the 1.78x deficit against rank-1 is mostly an
# artifact of running the wrong tile configuration.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd

echo "===== 1. how does eval.py parse the case line? ====="
grep -n -B4 -A12 'def _parse\|int(val)\|has_bias' "$SRC/eval.py" | head -60

echo
echo "===== 2. what does that do to 'False'? ====="
docker exec dhk-gemmrs python3 -c "
val = 'False'
try:
    parsed = int(val)
except ValueError:
    parsed = val
print('parsed =', repr(parsed), ' truthiness =', bool(parsed))
"

echo
echo "===== 3. does generate_input actually build a bias then? ====="
grep -n -B3 -A14 'def generate_input' "$SRC/task.py" 2>/dev/null | head -40
grep -n 'has_bias' "$SRC/task.py" "$SRC/reference.py" 2>/dev/null | head -20

echo
echo "===== 4. which config does OUR resolver pick, with and without bias? ====="
docker exec -w $ON/harness dhk-gemmrs python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('dhk_rt', 'build/dhk_rt.so')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
shapes = [(64,7168,18432,False),(512,4096,12288,True),(2048,2880,2880,True),
          (4096,4096,4096,False),(8192,4096,14336,True),(8192,8192,29568,False)]
print(f\"{'shape':>26} {'bias':>6} {'row':>4} {'BM/BN/BK':>12} {'NR':>4} {'tiles':>8}\")
for (mm,nn,kk,b) in shapes:
    for bias in (b, True):
        p = m.resolve_shape(mm,nn,kk,bias)
        tiles = p['gemm_tiles']
        tag = f'{mm}x{nn}x{kk}'
        cfg = f\"{p['bm']}/{p['bn']}/{p['bk']}\"
        print(f'{tag:>26} {str(bias):>6} {p[\"config_row\"]:>4} {cfg:>12} {p[\"num_reducer_ctas\"]:>4} {tiles:>8}')
    print()
" 2>&1 | tail -40

echo "===== DONE ====="
