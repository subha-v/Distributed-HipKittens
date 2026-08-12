#!/usr/bin/env bash
# Read-only: live (non-zombie) processes only, plus lease-holder liveness.
# The previous probe was flooded by <defunct> entries -- zombies matched the
# pattern and consumed the output budget, hiding whatever was actually running.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
LOCK=$ON/aug11/.gpu_lease

echo "===== lease holder liveness ====="
if [ -d "$LOCK" ]; then
  p=$(cat "$LOCK/pid" 2>/dev/null)
  who=$(cat "$LOCK/owner" 2>/dev/null)
  echo "owner=$who pid=$p"
  if [ -d "/proc/$p" ]; then
    echo "  holder ALIVE: $(ps -o stat=,etime=,cmd= -p "$p" 2>/dev/null | cut -c1-160)"
  else
    echo "  *** HOLDER PROCESS IS GONE -- the lease is ORPHANED ***"
  fi
else
  echo "lease FREE"
fi

echo
echo "===== live processes under our tree (zombies excluded) ====="
ps -eo pid,stat,etime,cmd --no-headers 2>/dev/null \
  | awk '$2 !~ /^Z/' \
  | grep -E 'aug11|dhk/distributed-kernels|ladder|eval\.py|trace_run|rocprof|torchrun|hipcc' \
  | grep -v grep | cut -c1-190 | head -25
echo "  (blank = nothing of ours is running)"

echo
echo "===== zombie census (informational) ====="
echo "  defunct python3 count: $(ps -eo stat=,cmd= --no-headers 2>/dev/null | awk '$1 ~ /^Z/' | grep -c python3)"
echo "  their parents:"
ps -eo ppid=,stat= --no-headers 2>/dev/null | awk '$2 ~ /^Z/ {print $1}' | sort | uniq -c | sort -rn | head -5

echo
echo "===== docker-side live processes ====="
docker exec dhk-gemmrs bash -lc "ps -eo pid,stat,etime,cmd --no-headers | awk '\$2 !~ /^Z/' | grep -Ev 'ps -eo|awk|bash -lc' | tail -15" 2>&1 | tail -16

echo done
