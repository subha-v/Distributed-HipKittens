#!/usr/bin/env bash
# Free the lease so the LAST outstanding figure (exp_22 / Fig 3) can run.
#
# Situation: exp_24 finished shape 2 of the ladder (354 s, all five arms) and is
# now looping in its OWN drain check -- "waiting for kfd drain (fds=1)" -- on the
# dead m9 process. That corpse cannot be cleared and cannot be signalled, so the
# wait never terminates. exp_22, exp_21 and exp_26 are all queued behind the
# lease it holds.
#
# Priority call: exp_22 is the last missing paper figure and its runner relies on
# the ALREADY-FIXED gpu_lease.sh, so it will proceed the moment the lease frees.
# exp_24 needs a patched runner regardless, so its restart costs only the 354 s
# of shape 2, which is preserved in its log.
#
# Only plain bash is signalled -- no GPU process is touched. SIGTERM only.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
. "$ON/tools/kfd_live.sh"

echo "===== liveness accounting (the point of the fix) ====="
echo "  live KFD pids : $(kfd_live_count)"
echo "  stale (dead)  : $(kfd_stale_list)"

echo
echo "===== stop exp_24's non-terminating drain loop (bash only) ====="
for pat in 'exp_24_ladders/full_runner.sh' 'exp_24_ladders/run_ladders.sh'; do
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    echo "  SIGTERM $p  ($pat)"
    kill -TERM "$p" 2>/dev/null || true
  done
done
sleep 6
echo "  survivors:"
pgrep -af 'exp_24_ladders/(full_runner|run_ladders)' 2>/dev/null || echo "  (none)"

echo
echo "===== release the lease exp_24 was holding ====="
bash "$ON/tools/gpu_lease.sh" release exp_24 2>/dev/null || \
  bash "$ON/tools/gpu_lease.sh" steal orchestrator "exp_24 stuck in a non-terminating drain wait on a dead pid; freeing for exp_22 (last figure)"
bash "$ON/tools/gpu_lease.sh" status

echo
echo "===== preserve exp_24's completed shape-2 result ====="
L=$ON/aug11/exp_24_ladders/logs/full_run.log
if [ -f "$L" ]; then
  cp -f "$L" "$ON/aug11/exp_24_ladders/logs/full_run.partial_shape2.log"
  echo "  saved -> logs/full_run.partial_shape2.log"
  grep -cE 'RANK-0|best=' "$L" 2>/dev/null | sed 's/^/  result lines: /'
fi

echo
echo "===== exp_22 should now acquire within ~15s ====="
sleep 20
bash "$ON/tools/gpu_lease.sh" status
pgrep -af 'exp_22_timeline/go_gpu.sh|gpu_lease.sh acquire exp_22' 2>/dev/null | head -3 || echo "  (exp_22 waiter not visible)"
echo done
