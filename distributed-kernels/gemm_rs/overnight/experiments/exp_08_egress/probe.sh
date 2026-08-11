#!/usr/bin/env bash
# exp_08 step 0: node cleanliness + profiler availability + counter inventory.
# Read-only. No GPU job is launched.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/experiments/exp_08_egress

echo "=== M0: KFD pids (must be empty) ==="
rocm-smi --showpids 2>/dev/null | sed -n '1,40p'

echo
echo "=== clocks ==="
rocm-smi --showclocks 2>/dev/null | grep -iE 'sclk|GPU\[' | sed -n '1,24p'

echo
echo "=== rocprofv3 presence ==="
docker exec dhk-gemmrs bash -lc 'which rocprofv3 rocprof rocprofv2 2>/dev/null; rocprofv3 --version 2>&1 | head -5'

echo
echo "=== counter inventory: TCC/EA write + read request families on gfx942 ==="
docker exec dhk-gemmrs bash -lc \
  'rocprofv3 --list-avail 2>/dev/null > /tmp/avail.txt; wc -l /tmp/avail.txt'
docker exec dhk-gemmrs bash -lc \
  'grep -oE "TCC_EA[0-9]*_(WR|RD)REQ[A-Z0-9_]*" /tmp/avail.txt | sort -u'
echo "--- derived metrics mentioning remote/fabric/xgmi ---"
docker exec dhk-gemmrs bash -lc \
  'grep -inE "remote|xgmi|fabric|infinity" /tmp/avail.txt | head -40'
echo "--- TCC_ counters (all, names only) ---"
docker exec dhk-gemmrs bash -lc \
  'grep -oE "\bTCC_[A-Z0-9_]+\b" /tmp/avail.txt | sort -u | tr "\n" " "'
echo
echo "--- copy the inventory out for the record ---"
docker exec dhk-gemmrs bash -lc "cp /tmp/avail.txt $OUT/rocprofv3_list_avail.txt 2>/dev/null; ls -l $OUT/"
