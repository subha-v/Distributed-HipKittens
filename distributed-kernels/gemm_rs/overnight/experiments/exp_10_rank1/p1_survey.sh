#!/usr/bin/env bash
# exp_10 probe 1 -- NO GPU. Establish the ground truth before spending a single
# GPU second: frozen-submission identity, iris staging state, and whether
# repair #5 (the hipIpcMemHandle_t alias) actually resolves.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
EXPECT=7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5

echo "===== 1. frozen submission identity ====="
ls -la "$RANK1" 2>&1
sha256sum "$RANK1" 2>&1
echo "expected $EXPECT"

echo
echo "===== 2. container python / triton / torch ====="
docker exec dhk-gemmrs bash -lc 'python3 -VV; python3 -c "import sys;print(sys.path)"'
docker exec dhk-gemmrs bash -lc 'python3 -c "import triton;print(\"triton\",triton.__version__)" 2>&1 | tail -2'
docker exec dhk-gemmrs bash -lc 'python3 -c "import torch;print(\"torch\",torch.__version__,\"devs\",torch.cuda.device_count())" 2>&1 | tail -2'

echo
echo "===== 3. iris staging at rank-1's hardcoded path ====="
docker exec dhk-gemmrs bash -lc 'ls -la /usr/local/lib/python3.10/dist-packages/iris/ 2>&1 | head -20'
echo "--- is it a real checkout or a symlink? ---"
docker exec dhk-gemmrs bash -lc 'ls -la /usr/local/lib/python3.10/dist-packages/ 2>&1 | head -20'

echo
echo "===== 4. iris checkouts available on the node ====="
ls -d /home/subvadla/iris* /home/subvadla/*/iris 2>/dev/null | head -20
find /home/subvadla -maxdepth 4 -name 'iris' -type d 2>/dev/null | head -20

echo
echo "===== 5. does iris import UNPATCHED, and does repair #5 resolve? ====="
docker exec -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages dhk-gemmrs bash -lc '
python3 -u -c "
import sys
try:
    import iris
    print(\"iris import OK\", iris.__file__)
except Exception as e:
    print(\"iris import FAILED:\", type(e).__name__, e)
    sys.exit(0)
try:
    import iris.hip as hip
    print(\"iris.hip import OK\", hip.__file__)
except Exception as e:
    print(\"iris.hip import FAILED:\", type(e).__name__, e)
    sys.exit(0)
for n in (\"hipIpcMemHandle_t\",\"gpuIpcMemHandle_t\",\"cudaIpcMemHandle_t\"):
    print(\"  has\", n, \"=\", hasattr(hip, n))
print(\"  ipc-ish symbols:\", sorted(x for x in dir(hip) if \"pc\" in x.lower())[:40])
"
' 2>&1

echo
echo "===== 6. same check WITH the compat sitecustomize on PYTHONPATH ====="
docker exec -e PYTHONPATH="$ON/tools/compat:/usr/local/lib/python3.10/dist-packages" -e COMPAT_SHIM_VERBOSE=1 dhk-gemmrs bash -lc '
python3 -u -c "
import iris.hip as hip
print(\"after shim: hipIpcMemHandle_t =\", hasattr(hip,\"hipIpcMemHandle_t\"))
try:
    h = hip.hipIpcMemHandle_t()
    print(\"  instantiated OK ->\", type(h))
except Exception as e:
    print(\"  instantiation FAILED:\", type(e).__name__, e)
"
' 2>&1

echo
echo "===== 7. previous attempt artifacts (.node, read-only) ====="
ls -la /home/subvadla/dhk/.node/compbench/rank1/ 2>&1 | head -30
echo "--- heap_bases anywhere on the node ---"
find /home/subvadla -name 'heap_bases_*.pkl' 2>/dev/null | head -20
echo "(none printed above == still zero)"

echo
echo "===== 8. staging root for this experiment ====="
ls -la "$ON/compbench/" 2>&1 | head
echo "===== DONE p1 ====="
