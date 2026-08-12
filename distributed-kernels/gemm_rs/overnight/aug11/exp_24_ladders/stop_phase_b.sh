#!/usr/bin/env bash
# Stop phase B gracefully. It is producing MISLABELLED captures: every arm lands in
# rot0/rank1 (see the local-builtin expansion bug), so the cross-check would report
# our own numbers as rank-1's. Better to lose ten minutes of the reference arm than
# to publish that.
#
# SIGTERM only, never SIGKILL: this kernel uses HIP IPC and leaked mappings wedge
# the node. Order matters -- quiesce the in-container eval.py first, because killing
# the `docker exec` client alone would detach and leave the ranks running.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "===== before ====="
pgrep -af 'full_runner.sh|run_ladders.sh|eval\.py' | grep -v pgrep || echo "nothing running"

echo
echo "===== 1. SIGTERM eval.py inside the container (graceful) ====="
docker exec dhk-gemmrs pkill -TERM -f 'eval.py' 2>&1 || echo "  (no eval.py in container)"
sleep 15

echo
echo "===== 2. SIGTERM the runner's session group ====="
for p in $(pgrep -f 'full_runner.sh' | head -3); do
  pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
  echo "  runner pid $p pgid $pgid -> kill -TERM -$pgid"
  [ -n "$pgid" ] && kill -TERM -"$pgid" 2>/dev/null || true
done
sleep 20

echo
echo "===== 3. settle ====="
for i in $(seq 1 18); do
  n=$(pgrep -cf 'full_runner.sh|run_ladders.sh|eval\.py' 2>/dev/null || echo 0)
  [ "$n" = "0" ] && { echo "  all stopped after $((i*5-5))s"; break; }
  [ "$i" = "1" ] && echo "  waiting for $n process(es) to exit"
  sleep 5
done

echo
echo "===== after ====="
pgrep -af 'full_runner.sh|run_ladders.sh|eval\.py' | grep -v pgrep || echo "nothing running"
bash "$D/toolsnap/gpu_lease.sh" status
echo "kfd fds: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"

echo
echo "===== lease released? if still held by exp_24, release it ====="
if bash "$D/toolsnap/gpu_lease.sh" status | grep -q "HELD by 'exp_24'"; then
  bash "$D/toolsnap/gpu_lease.sh" release exp_24
fi
bash "$D/toolsnap/gpu_lease.sh" status

echo
echo "===== discard the mislabelled captures ====="
rm -rf "$D/raw/eval"
echo "raw/eval removed; instrument A samples untouched:"
for i in 0 1 2 3 4 5; do
  echo "  shape idx $i : $(ls "$D/raw/ladder"/lad_s${i}.rank*.json 2>/dev/null | wc -l)/8"
done
