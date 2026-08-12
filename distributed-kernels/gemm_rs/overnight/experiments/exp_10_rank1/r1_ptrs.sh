#!/usr/bin/env bash
# exp_10 -- (a) read rank-1's dist_barrier / init C++ (the first peer write),
# (b) dump the IPC heap pointer table and its page permissions on all 8 ranks.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
PORT=${1:-12388}

echo "===== A. the C++: what is d_tensor, and what does barrier() write? ====="
awk 'NR>=356 && NR<=400 {printf "%d: %s\n", NR, $0}' "$RANK1"
echo "--- dist_barrier + the launcher + pybind defs (456..610) ---"
awk 'NR>=456 && NR<=530 {printf "%d: %s\n", NR, $0}' "$RANK1"
echo "..."
awk 'NR>=560 && NR<=610 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== B. GPU: pointer table + page permissions ====="
echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
cp "$ON/experiments/exp_10_rank1/r1_ptrs.py" "$ARM/r1_ptrs.py"
LOGS=$ARM/logs_ptrs
rm -rf "$LOGS"; mkdir -p "$LOGS"
rm -f "$ARM"/ipc_handles_rank*.bin

docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e R1_OUTDIR="$LOGS" \
  -e HSA_ENABLE_COREDUMP=0 \
  dhk-gemmrs bash -lc "timeout --signal=TERM 300 setsid python3 -u r1_ptrs.py $PORT" \
  > "$LOGS/driver.txt" 2>&1
echo "rc=$?"

echo
echo "--- driver summary ---"
tail -15 "$LOGS/driver.txt"

echo
echo "===== rank 0 pointer table ====="
cat "$LOGS/ptr_stdout_rank0.txt" 2>/dev/null

echo
echo "===== rank 1 pointer table ====="
cat "$LOGS/ptr_stdout_rank1.txt" 2>/dev/null

echo
echo "===== any rank stderr with content ====="
for r in 0 1 2 3 4 5 6 7; do
  f="$LOGS/ptr_stderr_rank$r.txt"
  [ -s "$f" ] && { echo "--- rank $r ---"; tail -20 "$f"; }
done
echo "===== DONE ====="
