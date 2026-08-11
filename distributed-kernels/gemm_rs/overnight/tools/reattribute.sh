#!/usr/bin/env bash
# Re-run the macro-gated stage attribution on the CURRENT winner.
#
# Necessary because the original attribution (GEMM 1317.8 / egress 919.7 /
# release 250.3 / sync 246.9 / reduce 219.5 us on shape 6) was measured on a
# binary that had neither exp_02's NR=32 nor exp_03's mainloop overlap. Its
# shape-6 total of 2861.7 us matches the OLD NR=8 split, not today's 2520 us.
# Choosing between E2 and E3 off those numbers would be choosing off a kernel
# that no longer exists.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$ON/experiments/logs
mkdir -p "$L"
stamp=$(date -u +%Y%m%dT%H%M%SZ)

n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || { echo "ABORT: node dirty"; exit 1; }

echo "===== stage attribution on the current winner ====="
docker exec -w $ON/harness dhk-gemmrs timeout 2400 \
  python3 -u exp_ablation.py 40 2>&1 | tee "$L/reattribute_$stamp.log"

echo "===== log: $L/reattribute_$stamp.log ====="
