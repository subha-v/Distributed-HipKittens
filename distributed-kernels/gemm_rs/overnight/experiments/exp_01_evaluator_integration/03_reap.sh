#!/usr/bin/env bash
# `timeout` only SIGTERMs the direct child, so mp_smoke's 8 spawned ranks can
# outlive it as orphans holding GPU memory and IPC mappings. SIGTERM only --
# SIGKILL on a HIP IPC process can wedge the node.
set -uo pipefail

echo "===== stragglers ====="
pgrep -a -f 'mp_smoke|eval.py|from multiprocessing.spawn' 2>&1 | head -30 || echo none

echo
echo "===== SIGTERM them ====="
pkill -TERM -f 'mp_smoke' 2>&1; echo "pkill mp_smoke rc=$?"
pkill -TERM -f 'multiprocessing.spawn' 2>&1; echo "pkill spawn rc=$?"
sleep 8

echo
echo "===== survivors ====="
pgrep -a -f 'mp_smoke|multiprocessing.spawn' 2>&1 | head -20 || echo none

echo
echo "===== rocm-smi pids (must be empty) ====="
rocm-smi --showpids 2>&1 | tail -8

echo
echo "===== free memory ====="
rocm-smi --showmemuse 2>&1 | grep -i -m9 'GPU\[' || true

echo
echo "===== REAP DONE ====="
