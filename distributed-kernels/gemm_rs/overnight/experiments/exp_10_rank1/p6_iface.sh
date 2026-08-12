#!/usr/bin/env bash
# exp_10 probe 6 -- NO GPU. Get the exact input/output contract so we can drive
# rank-1's custom_kernel from a minimal 8-rank driver of our own, instead of
# through eval.py (25 min per opaque timeout).
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== A. task.py ====="
cat "$ARM/task.py"

echo
echo "===== B. reference.py ====="
cat "$ARM/reference.py"

echo
echo "===== C. utils.py -- just the helpers ====="
grep -nE '^def |^class |^[A-Za-z_]+ *=' "$ARM/utils.py"

echo
echo "===== D. eval.py -- how a rank is set up and custom_kernel called ====="
grep -nE '^def |^class |init_process_group|set_device|custom_kernel|generate_input|check_implementation|destroy_process_group|Pool|\.get\(' "$ARM/eval.py"

echo
echo "--- _run_distributed_benchmark / the worker body (the traceback frame) ---"
awk 'NR>=290 && NR<=350 {printf "%d: %s\n", NR, $0}' "$ARM/eval.py"

echo
echo "===== E. rank-1 custom_kernel + patch_destroy_process_group (1738..1810) ====="
awk 'NR>=1738 && NR<=1810 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== F. is hipIpcMemHandle_t NATIVE in iris 4df4e85f? (repair #5 status) ====="
docker exec -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages dhk-gemmrs bash -lc '
python3 -u -S -c "
import iris.hip as hip
print(\"NO sitecustomize (-S). native hipIpcMemHandle_t =\", hasattr(hip,\"hipIpcMemHandle_t\"))
print(\"native gpuIpcMemHandle_t  =\", hasattr(hip,\"gpuIpcMemHandle_t\"))
"' 2>&1 | tail -5

echo
echo "===== G. our interleaved graded harness -- arm structure ====="
grep -nE '^def |^class |^ARMS|arm|import ' "$ON/experiments/exp_07_cold_l2/mp_graded.py" | head -50
echo "===== DONE p6 ====="
