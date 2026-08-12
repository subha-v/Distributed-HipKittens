#!/usr/bin/env bash
# The node is busy: exp_24 holds the lease with exp_21 and exp_26 queued. Our
# job is waiting its turn, which is correct -- but its outer `timeout 5400` was
# sized for the RUN, not for the run plus an unknown queue wait, so it could be
# killed mid-capture. Re-arm it with a window that covers both.
#
# Killing a WAITING acquirer is safe: it holds nothing. And if it wins the lease
# between the check and the signal, go_gpu.sh's EXIT/TERM trap releases it. We
# send TERM only, never KILL, and only ever to our own exp_22 processes.
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E22=$ON/aug11/exp_22_timeline

echo "===== CPU-only: validate the arm (a) fix against the captured trace ====="
CSV=$(ls -S "$E22"/prof/s5/*kernel_trace*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
  echo "replaying $CSV"
  docker exec -w "$E22" dhk-gemmrs python3 -u "$E22/b0_kernel_map.py" \
    --trace-dir "$E22/prof/s5" --boundaries "$E22/prof/s5/boundaries.json" \
    --map-json "$E22/b0_kernel_map.json" \
    --events-json "$E22/events_b0_reference.json" \
    --shape 8192,4096,14336,1,7168
  echo "replay rc=$?"
else
  echo "no captured trace on disk; the fix will be exercised by the queued run"
fi

echo
echo "===== re-arm the queued exp_22 job with a longer outer window ====="
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -4
for pid in $(pgrep -f "exp_22_timeline/go_gpu.sh"); do
  echo "TERM -> $pid (our own waiting job)"; kill -TERM "$pid" 2>/dev/null
done
pkill -TERM -f "gpu_lease.sh acquire exp_22" 2>/dev/null
sleep 3
LOG=$E22/logs/go_gpu.log
: > "$LOG"
setsid timeout 14400 env EXP22_REUSE_M7=1 \
  bash "$E22/go_gpu.sh" all >> "$LOG" 2>&1 < /dev/null &
sleep 4
echo "requeued pid $!"
bash "$ON/tools/gpu_lease.sh" status 2>&1 | head -4
