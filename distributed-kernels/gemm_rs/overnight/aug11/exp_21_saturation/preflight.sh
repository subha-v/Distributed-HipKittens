#!/usr/bin/env bash
# CPU-only pre-flight: rebuild the module so it provably matches the pushed source,
# byte-compile the driver, identify anything already on the GPUs, report the lease.
# Touches no GPU.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/aug11/exp_21_saturation

echo "########## driver syntax ##########"
docker exec dhk-gemmrs python3 -m py_compile "$EXP/run_saturation.py" && echo "  py_compile OK"

echo
echo "########## rebuild (the scoped push reset the source mtime) ##########"
docker exec dhk-gemmrs bash "$EXP/build.sh" 2>&1 | tail -25

echo
echo "########## import (no kernel launch) ##########"
docker exec dhk-gemmrs python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('sat_ubench', '$EXP/build/sat_ubench.so')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('  GRID_MAX =', m.GRID_MAX, ' devices =', m.device_count())
" 2>&1 | tail -5

echo
echo "########## lease ##########"
bash "$ON/tools/gpu_lease.sh" status
tail -6 "$ON/aug11/gpu_lease.log" 2>/dev/null

echo
echo "########## what is on the GPUs, and whose is it ##########"
rocm-smi --showpids 2>&1 | grep -E '^[0-9]+' || echo "  (no KFD pids)"
for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do
  echo "-- host pid $p --"
  ps -o pid,etimes,user,cmd -p "$p" 2>/dev/null | tail -2
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-300; echo
done
echo
echo "-- our known GPU drivers alive? --"
ps -eo pid,etimes,cmd 2>/dev/null \
  | grep -E 'run_saturation|sweep\.py|ladder|m7_bench|exp_ablation|mp_smoke|eval\.py|campaign' \
  | grep -v grep || echo "  (none)"
