#!/usr/bin/env bash
# exp_08 step 1a: does rocprofv3 counter collection survive an 8-rank
# single-process run of THIS protocol at all?
#
# The risk is specific and it invalidates everything downstream if ignored:
# counter collection inserts barriers around dispatches. If it serializes the
# eight agents, rank 0's reducers spin on ready flags that no other rank has
# published yet, the bounded wait times out, the sticky error bit is set, and
# every producer returns EARLY WITHOUT EMITTING -- which would deflate every
# traffic counter to near zero and look like a wonderful result.
#
# Gate: PROF line must read correct=1 tight=1 errors=none.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/experiments/exp_08_egress
mkdir -p "$D/prof"

echo "=== node clean check ==="
n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
echo "KFD pids: $n"
[ "$n" = "0" ] || { echo "ABORT: node dirty"; exit 1; }

echo
echo "=== which modules are already built ==="
docker exec dhk-gemmrs bash -lc "ls -l $ON/harness/build/*.so | sed 's#.*/##'"

echo
echo "=== 1a: unprofiled control run (proves the driver itself is sound) ==="
docker exec -w $D dhk-gemmrs timeout 600 \
  python3 -u prof_driver.py gemm_rs_mi300x 8192 8192 29568 0 42 2 4 2>&1 | tail -6

echo
echo "=== 1b: same run under rocprofv3, 4 counters ==="
rm -rf "$D/prof/smoke"
docker exec -w $D dhk-gemmrs timeout 900 \
  rocprofv3 --pmc TCC_EA0_WRREQ TCC_EA0_WRREQ_64B TCC_EA0_WRREQ_DRAM TCC_EA0_RDREQ \
    -d "$D/prof/smoke" -o s --output-format csv \
    -- python3 -u prof_driver.py gemm_rs_mi300x 8192 8192 29568 0 42 2 4 2>&1 \
  | grep -viE '^\s*$' | tail -25

echo
echo "=== csv layout ==="
find "$D/prof/smoke" -name '*.csv' | while read -r f; do
  echo "--- $f ($(wc -l < "$f") lines)"
  head -2 "$f"
done
