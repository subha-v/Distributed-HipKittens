#!/usr/bin/env bash
# Poll the detached rematch run: is it alive, how far has it got, what do the
# finished shapes say.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1

echo "alive: $(pgrep -c -f 'rematch_sweep.sh|rematch_launch.sh|mp_vs_rank1' || true) procs"
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
echo "shapes with rank0 json: $(ls "$E/vs_logs"/vs_s*.rank0.json 2>/dev/null | wc -l)"
echo
echo "===== tail of rematch_run.log ====="
tail -n "${1:-60}" "$E/rematch_run.log" 2>&1
echo "===== DONE ====="
