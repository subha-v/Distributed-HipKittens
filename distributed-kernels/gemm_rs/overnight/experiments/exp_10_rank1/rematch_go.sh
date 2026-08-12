#!/usr/bin/env bash
# Detach rematch_launch.sh from this ssh session and return immediately.
set -u
E=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/experiments/exp_10_rank1
SHAPES=${1:-5,3,0,1,2,4}
ITERS=${2:-12}
REPS=${3:-2}

if pgrep -f 'rematch_launch.sh|rematch_sweep.sh' > /dev/null; then
  echo "REFUSING: a rematch run is already alive"
  pgrep -af 'rematch_launch.sh|rematch_sweep.sh'
  exit 1
fi

setsid nohup bash "$E/rematch_launch.sh" "$SHAPES" "$ITERS" "$REPS" \
  < /dev/null > /dev/null 2>&1 &
echo "detached pid=$!  shapes=$SHAPES iters=$ITERS reps=$REPS"
sleep 3
echo "--- first lines ---"
head -5 "$E/rematch_run.log" 2>&1
echo "===== DONE ====="
