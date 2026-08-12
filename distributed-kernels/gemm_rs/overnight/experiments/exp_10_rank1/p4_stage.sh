#!/usr/bin/env bash
# exp_10 probe 4 -- NO GPU. (a) find rank-1's symmetric-heap sizing in its C++
# extension, (b) re-stage iris at the hardcoded python3.10 path (the container
# was recreated and lost it), (c) stage + patch the arm and diff the patch.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
ARM=$ON/compbench/rank1
IRISSRC=/home/subvadla/amd-master/iris
IRISDST=/usr/local/lib/python3.10/dist-packages

echo "===== A. heap sizing / IpcCommBlock / hipMalloc in the C++ ====="
grep -nE '#define|RANK_SIZE|struct IpcCommBlock|hipMalloc|MAX_|SIZE|BUFFER|constexpr' "$RANK1" | awk 'NR<80'

echo
echo "--- IpcCommBlock struct verbatim ---"
awk 'NR>=318 && NR<=345 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "--- the init / allocate path ---"
awk 'NR>=396 && NR<=455 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== B. re-stage iris at rank-1's hardcoded path (needs root: dhk-eval) ====="
echo "--- source checkout ---"
ls -la "$IRISSRC/iris/" | head -20
echo "--- iris git revision ---"
git -C "$IRISSRC" rev-parse --short HEAD 2>/dev/null || echo "(not a git repo)"
echo "--- staging ---"
docker exec dhk-eval bash -lc "mkdir -p $IRISDST && rm -rf $IRISDST/iris && cp -r $IRISSRC/iris $IRISDST/iris && chmod -R a+rX $IRISDST/iris && ls $IRISDST/iris | head"
echo "--- lines 60..90 of the staged iris __init__.py (what rank-1 wants to comment out) ---"
docker exec dhk-eval bash -lc "awk 'NR>=60 && NR<=90 {printf \"%d: %s\n\", NR, \$0}' $IRISDST/iris/__init__.py"

echo
echo "===== C. sudo shim (repair #2): a no-op sudo early on PATH ====="
mkdir -p "$ON/tools/compat/bin"
printf '#!/bin/sh\n# repair #2: rank-1 runs `sudo sed -i \x2766,82 s/^/#/\x27 .../iris/__init__.py`.\n# In this iris revision those lines are the `from . import hip` block, so the\n# real sed would delete the very iris.hip the submission then calls. No-op.\nexit 0\n' > "$ON/tools/compat/bin/sudo"
chmod +x "$ON/tools/compat/bin/sudo"
cat "$ON/tools/compat/bin/sudo"

echo
echo "===== D. does iris import now, unpatched, and is repair #5 moot or needed? ====="
docker exec -e PYTHONPATH="$IRISDST" dhk-gemmrs bash -lc '
python3 -u -c "
import iris, iris.hip as hip
print(\"iris OK      :\", iris.__file__)
print(\"iris.hip OK  :\", hip.__file__)
for n in (\"hipIpcMemHandle_t\",\"gpuIpcMemHandle_t\"):
    print(\"  has\", n, \"=\", hasattr(hip, n))
"' 2>&1

echo
echo "===== E. stage the arm and patch it ====="
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
ls -la "$SRC" | head -12
rm -rf "$ARM"; mkdir -p "$ARM"
cp "$SRC/eval.py" "$SRC/task.py" "$SRC/utils.py" "$SRC/reference.py" "$ARM/"
cp "$SRC/cases.txt" "$ARM/cases_test.txt" 2>/dev/null || true
python3 "$ON/tools/patch_rank1.py" "$RANK1" "$ARM/submission.py"
echo "rc=$?"
echo
echo "--- diff frozen vs staged (should be exactly the packed_metadata repair) ---"
diff -u "$RANK1" "$ARM/submission.py"
echo "(diff rc=$? ; 1 == differs, expected)"

echo
echo "===== F. arm contents ====="
ls -la "$ARM"
echo "===== DONE p4 ====="
