#!/usr/bin/env bash
# rank-1 replaces triton's HIPLauncher. Find out how Triton 3.6.0 invokes the
# launcher, and what rank-1's generated C `launch` expects, so the Python-level
# calling convention can be adapted without giving up their launcher.
set -uo pipefail
NAME=dhk-eval
R=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
run() { docker exec "$NAME" bash -c "$1"; }

echo "===== triton 3.6.0 stock HIPLauncher ====="
run 'python3 - <<PY
import inspect
from triton.backends.amd import driver
src = inspect.getsource(driver.HIPLauncher)
print(src)
PY'

echo
echo "===== who calls the launcher, and with what ====="
run 'grep -n "self.launch\|launcher(\|def __call__\|launch_metadata\|launch_enter_hook" /usr/local/lib/python3.12/dist-packages/triton/backends/amd/driver.py | head -30'
echo "--- CompiledKernel.__call__ / run in triton.compiler ---"
run 'grep -n -A22 "def run" /usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py | head -60'

echo
echo "===== the tail of triton 3.6 make_launcher generated signature ====="
run 'grep -n -B4 -A18 "_launch(PyObject\|static PyObject \*launch" /usr/local/lib/python3.12/dist-packages/triton/backends/amd/driver.py | head -60'

echo
echo "===== rank-1 make_launcher: what does ITS C launch take? ====="
grep -n -B6 -A30 'def make_launcher' "$R" | head -70
echo "--- its PyArg parsing / launch signature ---"
grep -n 'PyArg_ParseTuple\|static PyObject\* launch\|static PyObject \*launch\|args_new\|count_num' "$R" | head -30
