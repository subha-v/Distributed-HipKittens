#!/usr/bin/env bash
# The frozen rank-1 file imports symbols from triton.backends.amd.driver that
# this node's Triton may not have. Establish exactly which symbols are missing
# and how rank-1 uses them, before deciding on a repair.
set -uo pipefail
NAME=dhk-eval
R=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

run() { docker exec "$NAME" bash -c "$1"; }

echo "===== triton version in the eval container ====="
run 'python3 -c "import triton;print(triton.__version__, triton.__file__)"'

echo
echo "===== which symbols rank-1 imports from triton internals ====="
grep -nE 'from triton[.a-z_]* import|import triton' "$R" | head -20

echo
echo "===== do those symbols exist? ====="
run 'python3 - <<PY
import importlib
wanted = {
 "triton.backends.amd.driver": ["ty_to_cpp","_BASE_ARGS_FORMAT","FLOAT_STORAGE_TYPE",
   "FLOAT_PACK_FUNCTION","_get_path_to_hip_runtime_dylib","compile_module_from_src",
   "wrap_handle_tensor_descriptor","include_dirs"],
 "triton.runtime": ["_allocation"],
 "triton.compiler.errors": ["CompilationError"],
}
for mod, names in wanted.items():
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"{mod}: IMPORT FAILED {e}"); continue
    for n in names:
        print(f"  {mod}.{n}: {\"present\" if hasattr(m,n) else \"MISSING\"}")
PY'

echo
echo "===== how rank-1 uses wrap_handle_tensor_descriptor ====="
grep -n -B8 -A25 'wrap_handle_tensor_descriptor' "$R"

echo
echo "===== triton versions available in other images on this node ====="
for img in primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix \
           amdsiloai/amd-internal-gordon:vllm-v0.23.1rc1-09663ab-aiter-0.1.16.post2 \
           vllm/vllm-openai-rocm:v0.26.0; do
  v=$(timeout 180 docker run --rm --entrypoint /bin/bash "$img" -c \
      'python3 -c "import triton,inspect;from triton.backends.amd import driver;print(triton.__version__, hasattr(driver,\"wrap_handle_tensor_descriptor\"))" 2>/dev/null' 2>/dev/null)
  echo "  $img -> ${v:-<probe failed>}"
done
