#!/usr/bin/env bash
# exp_20: read-only pre-flight for the counter pass.
#   - who holds the GPUs (the poll reports 1 KFD pid with our job finished)
#   - the arm .so set that the attribution actually measured, with timestamps
#   - exp_08's driver signature, so its arg order is reused rather than guessed
#   - rocprofv3 presence/version and whether the gfx942 counters resolve
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E8=$ON/experiments/exp_08_egress

echo "===== KFD occupancy ====="
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
echo "--- ps for any of our python/eval workers ---"
ps -eo pid,user,etime,cmd 2>/dev/null | grep -E 'exp_ablation|prof_driver|m7_bench|eval' \
  | grep -v grep || echo "  none"

echo
echo "===== ablation arm .so set (must be minutes old, not Aug 11 15:31) ====="
ls -la --time-style=full-iso $ON/harness/build/gemm_rs_abl_*.so 2>&1
ls -la --time-style=full-iso $ON/harness/build/gemm_rs_mi300x.so 2>&1

echo
echo "===== exp_08 driver signature ====="
ls $E8 2>&1
sed -n '1,40p' $E8/prof_driver.py 2>&1

echo
echo "===== rocprofv3 ====="
docker exec dhk-gemmrs /opt/rocm/bin/rocprofv3 --version 2>&1 | head -5
echo "done"
