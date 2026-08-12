#!/usr/bin/env bash
# Regenerate the M9 golden against the CURRENT shape table, and prove the new
# staleness guard behaves.
#
# Why this was needed: the golden .so on the node predated exp_14's retile, so
# it computed a different tile map than the plan it was handed and read 192
# rows past the end of a 2880-row B operand on row 3 -- a GPU memory fault that
# wedged the node in driver teardown and blocked two other experiments. M9 is
# supposed to be the gate that protects publication order; a gate that faults
# the node is worse than no gate.
#
# Compile only. No GPU work here.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_05_release_granularity

echo "===== golden .so and sidecar BEFORE ====="
ls -la $ON/harness/build/gemm_rs_mi300x_e3base.* 2>/dev/null || echo "  (none)"

echo
echo "===== current shape table for M9's cases ====="
docker exec -w $ON/harness dhk-gemmrs python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('dhk_rt','build/dhk_rt.so')
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
for (m,n,k,b) in [(512,4096,12288,True),(8192,8192,29568,False),
                  (8192,8192,28672,False),(64,7168,18432,False),
                  (2048,2880,2880,True),(4096,4096,4096,False)]:
    p = rt.resolve_shape(m,n,k,b)
    print(f'  {m}x{n}x{k} bias={int(b)}: {p[\"bm\"]}/{p[\"bn\"]}/{p[\"bk\"]} row={p[\"config_row\"]}')
"

echo
echo "===== guard should REFUSE while the sidecar is missing/stale ====="
docker exec -w $ON/harness dhk-gemmrs timeout 120 python3 m9_stale_slot.py 0.01 2>&1 | head -20

echo
echo "===== rebuild the golden (compile only) ====="
docker exec dhk-gemmrs bash $E/build_golden.sh 2>&1 | tail -25

echo
echo "===== golden .so and sidecar AFTER ====="
ls -la $ON/harness/build/gemm_rs_mi300x_e3base.* 2>/dev/null
echo "--- sidecar contents ---"
docker exec dhk-gemmrs cat $ON/harness/build/gemm_rs_mi300x_e3base.table.json 2>/dev/null

echo "===== DONE ====="
