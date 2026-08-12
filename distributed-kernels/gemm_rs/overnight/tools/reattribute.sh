#!/usr/bin/env bash
# Re-run the macro-gated stage attribution on the CURRENT winner.
#
# Necessary because an attribution taken on an older binary describes a kernel
# that no longer exists, and ranking the next experiment off it is how a session
# gets spent optimizing the wrong pool.
#
# TRAP, paid twice: exp_ablation.py SKIPS compilation when the arm .so already
# exists ("already built"). After a kernel change that silently re-reports the
# PREVIOUS kernel's attribution -- caught only because `full` for shape 6 read
# 2527 us when the real kernel was 1829. This script now forces a rebuild and
# then ASSERTS that `full` matches the expected total, so a stale table cannot
# be mistaken for a fresh one.
#
#   reattribute.sh [expected_shape6_full_us] [tolerance_pct]
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
L=$ON/experiments/logs
EXPECT=${1:-0}
TOLPCT=${2:-8}
mkdir -p "$L"
stamp=$(date -u +%Y%m%dT%H%M%SZ)

# Wait for a clean node rather than aborting on the first sample. A previous
# run's last worker can linger for tens of seconds after its parent returns, and
# aborting on that wastes a whole invocation.
#
# Liveness-aware since 2026-08-12: the old version counted `rocm-smi --showpids`
# rows, which include processes that have CRASHED and wedged in `exit_mm`. Such a
# process has no address space, cannot dispatch a kernel, and cannot be
# signalled -- so waiting for it never terminates. That cost the aug11 figure
# queue ~40 minutes across several campaigns. See tools/kfd_live.sh.
. "$(dirname "${BASH_SOURCE[0]}")/kfd_live.sh"
kfd_wait_clean 30 10 || exit 1

echo "===== forcing a rebuild of every ablation arm ====="
docker exec dhk-gemmrs bash -c "rm -fv $ON/harness/build/gemm_rs_abl_*.so $ON/harness/ablate/*.cpp 2>/dev/null; true"

echo "===== stage attribution on the current winner ====="
docker exec -w $ON/harness dhk-gemmrs timeout 3600 \
  python3 -u exp_ablation.py 40 2>&1 | tee "$L/reattribute_$stamp.log"

if [ "$EXPECT" != "0" ]; then
  echo "===== freshness assertion ====="
  got=$(grep -E '^8192x8192x29568' "$L/reattribute_$stamp.log" | awk '{print $2}')
  echo "  shape 6 full = $got us, expected ~$EXPECT us (+/-${TOLPCT}%)"
  ok=$(awk -v g="$got" -v e="$EXPECT" -v t="$TOLPCT" \
        'BEGIN{d=(g-e)/e*100; if(d<0)d=-d; print (d<=t)?"yes":"no"}')
  if [ "$ok" != "yes" ]; then
    echo "  STALE: this table does not describe the current binary. Do not rank off it."
    exit 2
  fi
  echo "  fresh."
fi

echo "===== log: $L/reattribute_$stamp.log ====="
