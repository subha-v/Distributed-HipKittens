#!/usr/bin/env bash
# exp_10 probe 5 -- NO GPU. Repair #1 done right: rank-1 needs iris at TWO
# places for TWO different reasons, and dhk-eval's /usr/local/lib is a different
# filesystem from dhk-gemmrs's, so staging in dhk-eval never helped.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
IRISSRC=/home/subvadla/amd-master/iris/iris
IRISDST=/usr/local/lib/python3.10/dist-packages

echo "===== A. rank-1's module-level preamble (lines 1..37) ====="
awk 'NR>=1 && NR<=37 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== B. containers ====="
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1
echo "--- can we exec as root into OUR container? ---"
docker exec -u 0 dhk-gemmrs bash -lc 'id; echo ROOT_EXEC_OK' 2>&1 | tail -3

echo
echo "===== C. stage iris INSIDE dhk-gemmrs at the hardcoded path ====="
docker exec -u 0 dhk-gemmrs bash -lc "
  mkdir -p $IRISDST
  rm -rf $IRISDST/iris
  cp -r $IRISSRC $IRISDST/iris
  chmod -R a+rX $IRISDST/iris
  ls $IRISDST/iris
  echo '--- writable by uid 15523? (rank-1 only READS it; the sudo sed is shimmed) ---'
  ls -ld $IRISDST/iris $IRISDST/iris/__init__.py
" 2>&1

echo
echo "===== D. the two requirements, verified independently, as uid 15523 ====="
docker exec dhk-gemmrs bash -lc "
  echo '--- (1) the literal open() at submission.py:26 ---'
  python3 -c \"
p='$IRISDST/iris/__init__.py'
d=open(p).read()
print('open OK, bytes =', len(d))
print('lines 66..82 are:')
for i,l in enumerate(d.splitlines()[65:82], start=66):
    print('   ', i, l)
\"
" 2>&1

echo
echo "--- (2) import iris UNPATCHED, and repair #5's symbol ---"
docker exec -e PYTHONPATH="$ON/tools/compat:$IRISDST" -e COMPAT_SHIM_VERBOSE=1 dhk-gemmrs bash -lc '
python3 -u -c "
import iris, iris.hip as hip
print(\"iris     :\", iris.__file__)
print(\"iris.hip :\", hip.__file__)
for n in (\"hipIpcMemHandle_t\",\"gpuIpcMemHandle_t\"):
    print(\"  has\", n, \"=\", hasattr(hip,n))
import triton
from triton.backends.amd import driver
print(\"wrap_handle_tensor_descriptor present:\", hasattr(driver,\"wrap_handle_tensor_descriptor\"))
"' 2>&1

echo
echo "===== E. sudo shim reachable? ====="
docker exec -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin" dhk-gemmrs bash -lc 'which sudo; sudo sed -i "1s/^/X/" /nonexistent; echo "sudo shim rc=$?"' 2>&1

echo
echo "===== F. can rank-1's submission be IMPORTED end to end? (no GPU work yet) ====="
cd "$ON/compbench/rank1"
docker exec -w "$ON/compbench/rank1" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e COMPAT_SHIM_VERBOSE=1 \
  dhk-gemmrs bash -lc 'timeout 300 python3 -u -c "
import sys
sys.argv=[\"x\"]
import submission
print(\"SUBMISSION IMPORT OK\")
print(\"  CREATE_SHEMEM_CODE referenced anywhere at runtime? (dead string):\", len(submission.CREATE_SHEMEM_CODE))
"' 2>&1 | tail -40

echo "===== DONE p5 ====="
