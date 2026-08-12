#!/usr/bin/env bash
# Leave the node in a known, documented state and prove it: sources in sync,
# submission.py resynced, the built module current, and no GPU process alive.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight

echo "===== submission.py in sync with hk_submission.py ====="
cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" \
  && echo "  yes" || { echo "  NO -- resyncing"; \
       cp "$ON/harness/hk_submission.py" "$ON/harness/submission.py"; \
       cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" \
         && echo "  resynced" || echo "  RESYNC FAILED"; }

echo "===== LF-form hashes of the two files exp_12 changed ====="
sha256sum /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp \
          "$ON/harness/hk_submission.py"

echo "===== built module ====="
ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so"
# The spec name must be the module's own PYBIND11_MODULE name: the loader looks
# up PyInit_<name>, so loading it as anything else raises ImportError.
docker exec -w "$ON/harness" dhk-gemmrs python3 -c '
import importlib.util
spec = importlib.util.spec_from_file_location("gemm_rs_mi300x",
                                             "build/gemm_rs_mi300x.so")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("  entry points:", [x for x in dir(m) if not x.startswith("_")])
'

echo "===== node ====="
rocm-smi --showpids 2>&1 | grep -E 'No KFD PIDs|PID' | head -5
rocm-smi 2>&1 | tail -11 | head -9
echo "DONE-final"
