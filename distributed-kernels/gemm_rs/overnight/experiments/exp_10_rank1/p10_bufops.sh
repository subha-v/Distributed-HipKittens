#!/usr/bin/env bash
# exp_10 probe 10 -- NO GPU. Find the Triton 3.6.0 knob that disables the AMD
# "convert to buffer ops" pass. buffer_store's voffset is 32-bit, which is what
# truncates rank-1's i64 peer-heap offset.
set -uo pipefail

T=/usr/local/lib/python3.12/dist-packages/triton

echo "===== A. buffer-ops knobs in the python layer ====="
docker exec dhk-gemmrs bash -lc "
grep -rn -iE 'buffer_ops|bufferops|buffer-ops' $T --include=*.py | head -30
"

echo
echo "===== B. all AMD-related knobs / env vars Triton reads ====="
docker exec dhk-gemmrs bash -lc "
grep -rn -oE 'TRITON_[A-Z0-9_]+|AMDGCN_[A-Z0-9_]+' $T --include=*.py | sed 's/.*://' | sort -u | head -60
"

echo
echo "===== C. knobs.py: the amd section ====="
docker exec dhk-gemmrs bash -lc "
python3 -c \"
import triton, inspect
from triton import knobs
for name in dir(knobs):
    if name.startswith('_'): continue
    obj = getattr(knobs, name)
    if hasattr(obj, '__dict__') or 'amd' in name.lower():
        print('==', name, type(obj).__name__)
\" 2>&1 | head -30
"

echo
echo "--- the amd knob group in detail ---"
docker exec dhk-gemmrs bash -lc "
python3 -c \"
from triton import knobs
amd = getattr(knobs, 'amd', None)
print('knobs.amd =', amd)
if amd is not None:
    for k in dir(amd):
        if not k.startswith('_'):
            try: print('   ', k, '=', getattr(amd, k))
            except Exception as e: print('   ', k, 'ERR', e)
\" 2>&1 | head -40
"

echo
echo "===== D. the backend compiler: where is the buffer-ops pass invoked? ====="
docker exec dhk-gemmrs bash -lc "
grep -rn -iE 'buffer|use_buffer' $T/backends/amd/compiler.py | head -30
"

echo
echo "===== E. grep the compiled .so / libtriton for the pass option name ====="
docker exec dhk-gemmrs bash -lc "
ls $T/_C/ 2>/dev/null
strings $T/_C/libtriton.so 2>/dev/null | grep -iE 'buffer-ops|buffer_ops|convert-to-buffer|amdgpu-convert' | sort -u | head -20
"

echo
echo "===== F. AMD backend options dataclass (what the submission could set) ====="
docker exec dhk-gemmrs bash -lc "
python3 -c \"
from triton.backends.amd.compiler import HIPOptions
import dataclasses
for f in dataclasses.fields(HIPOptions):
    print(f'{f.name:35s} default={f.default!r}')
\" 2>&1 | head -50
"
echo "===== DONE p10 ====="
