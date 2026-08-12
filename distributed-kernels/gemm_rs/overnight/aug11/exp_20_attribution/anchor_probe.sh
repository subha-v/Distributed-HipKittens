#!/usr/bin/env bash
# Read-only: does exp_ablation.py's anchor set still apply to the CURRENT kernel?
# Prints one line per anchor with its occurrence count. count != 1 == broken.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
SRCD=/home/subvadla/dhk/distributed-kernels/gemm_rs

echo "===== md5 of the harness copies (node) ====="
md5sum $ON/harness/exp_ablation.py $ON/harness/exp_ablation_one.py 2>&1

echo
echo "===== node source, LF-normalized md5 ====="
for f in gemm_rs_mi300x.cpp gemm_rs_mi300x_hk_adapter.cuh \
         gemm_rs_mi300x_constants.cuh gemm_rs_mi300x_host_abi.hpp; do
  printf '%s  %s\n' "$(tr -d '\r' < $SRCD/$f | md5sum | cut -d' ' -f1)" "$f"
done

echo
echo "===== anchor occurrence counts ====="
python3 << 'PY'
import ast
ABL = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/exp_ablation.py"
SRC = "/home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp"
tree = ast.parse(open(ABL).read())
patches = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PATCHES":
        patches = ast.literal_eval(node.value)
src = open(SRC).read()
bad = 0
for i, (anchor, _) in enumerate(patches, 1):
    c = src.count(anchor)
    if c != 1:
        bad += 1
    first = anchor.strip().splitlines()[0][:72]
    print(f"  anchor {i}: count={c} {'OK    ' if c==1 else 'BROKEN'}  {first!r}")
print(f"  -> {bad} broken of {len(patches)}")
PY

echo
echo "===== ablate/ scratch state ====="
ls -la --time-style=full-iso $ON/harness/ablate/ 2>&1
for g in ABL_SKIP_MAINLOOP ABL_EMIT_LOCAL ABL_SKIP_REDUCE ABL_NO_PROTOCOL ABL_NO_RELEASE; do
  printf '   %-20s %s\n' "$g" "$(grep -c "$g" $ON/harness/ablate/gemm_rs_ablate.cpp 2>/dev/null || echo 0)"
done

echo
echo "===== where the per-shape NR / tile table lives ====="
grep -rn 'num_reducer_ctas' $SRCD/gemm_rs_mi300x_host_abi.hpp 2>&1 | head -20
echo "done"
