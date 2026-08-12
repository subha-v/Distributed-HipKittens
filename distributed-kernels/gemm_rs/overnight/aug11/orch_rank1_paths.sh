#!/usr/bin/env bash
# Read-only: establish which rank-1 staging paths actually exist, so the repair
# to tools/run_rank1_bench3.sh mirrors reality rather than its own comments.
# Runs no GPU work -- import checks only.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== 1. host-side compat tree ====="
ls -la "$ON/tools/compat" 2>&1
echo "-- bin? --"
ls -la "$ON/tools/compat/bin" 2>&1
echo "-- patch_rank1.py --"
ls -la "$ON/tools/patch_rank1.py" "$ON/patch_rank1.py" 2>&1

echo
echo "===== 2. frozen submission hash ====="
sha256sum "$RANK1" 2>&1
echo "expected prefix: 7940fcb8"

echo
echo "===== 3. per-container staging ====="
for C in dhk-gemmrs dhk-eval; do
  echo "--- container $C ---"
  docker exec "$C" bash -lc 'python3 -c "import sys; print(\"py\", sys.version.split()[0])" 2>&1' 2>&1
  docker exec "$C" bash -lc 'ls -d /usr/local/lib/python3.10/dist-packages/iris 2>&1' 2>&1
  docker exec "$C" bash -lc 'PYTHONPATH=/usr/local/lib/python3.10/dist-packages python3 -c "import iris, iris.hip; print(\"iris OK\", iris.__file__)" 2>&1 | tail -2' 2>&1
  docker exec "$C" bash -lc 'ls -la /usr/local/shim 2>&1 | head -5' 2>&1
  docker exec "$C" bash -lc 'which sudo; sudo --version 2>&1 | head -1' 2>&1
done

echo
echo "===== 4. sitecustomize importable via tools/compat ====="
docker exec dhk-gemmrs bash -lc "PYTHONPATH=$ON/tools/compat python3 -c 'import sitecustomize; print(\"sitecustomize OK\", sitecustomize.__file__)' 2>&1 | tail -2" 2>&1

echo
echo "===== 5. evaluator source tree ====="
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
ls -la "$SRC"/eval.py "$SRC"/task.py "$SRC"/utils.py "$SRC"/reference.py "$SRC"/cases.txt 2>&1

echo
echo "===== 6. existing compbench arms ====="
ls -la "$ON/compbench" 2>&1
echo "done"
