#!/usr/bin/env bash
set -uo pipefail
IRIS=/home/subvadla/amd-master/iris

echo "===== reference (GEMM+RCCL) arm result, if it finished ====="
sed -n '/# reference : eval.py benchmark/,$p' /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench.log 2>/dev/null | head -40

echo
echo "===== iris checkout layout ====="
ls -la "$IRIS" 2>&1 | head -25
echo "-- packaging --"
ls "$IRIS"/setup.py "$IRIS"/pyproject.toml "$IRIS"/setup.cfg 2>&1
echo "-- python package dir --"
find "$IRIS" -maxdepth 3 -name '__init__.py' 2>/dev/null | head

echo
echo "===== iris/__init__.py lines 55-95 (rank-1 comments out 66-82) ====="
INIT=$(find "$IRIS" -maxdepth 3 -path '*/iris/__init__.py' | head -1)
echo "init file: $INIT"
[ -n "$INIT" ] && sed -n '55,95p' "$INIT" | cat -n | sed 's/^/   /'

echo
echo "===== does iris/__init__.py import a compiled lib? ====="
[ -n "$INIT" ] && grep -nE 'import|ctypes|CDLL|\.so|hip' "$INIT" | head -25

echo
echo "===== iris.hip module (rank-1 uses iris.hip.get_ipc_handle / hipIpcMemHandle_t) ====="
[ -n "$INIT" ] && ls -la "$(dirname "$INIT")" | head -20
