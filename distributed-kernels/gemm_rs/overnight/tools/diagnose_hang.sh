#!/usr/bin/env bash
# The rank-1 run now fails only by timeout. Prime suspect: rank-1 calls
# iris.hip.hipIpcMemHandle_t(), which the iris revision staged here does NOT
# expose, inside the per-rank helper processes it spawns to build its symmetric
# heap. A helper dying there would leave the parent waiting.
set -uo pipefail
NAME=dhk-eval
DIR=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/rank1
R=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
run() { docker exec "$NAME" bash -c "$1"; }

echo "===== full stderr of the bench pass (their own diagnostics) ====="
run "wc -l $DIR/bench.stderr.txt $DIR/bench.stdout.txt 2>/dev/null"
run "head -60 $DIR/bench.stderr.txt 2>/dev/null"
echo "--- anything mentioning iris / ipc / shmem / rank in stdout ---"
run "grep -niE 'iris|ipc|shmem|heap|rank|Traceback|Error' $DIR/bench.stdout.txt 2>/dev/null | head -30"

echo
echo "===== what rank-1 needs from iris.hip ====="
grep -n 'iris\.hip\.[A-Za-z_]*' -o "$R" | sort -u

echo
echo "===== what the staged iris.hip actually provides ====="
run "grep -nE '^def |^class |^[A-Za-z_]+ *=' /usr/local/lib/python3.10/dist-packages/iris/hip.py | head -40"
echo "--- any Ipc symbols at all? ---"
run "grep -niE 'ipc' /usr/local/lib/python3.10/dist-packages/iris/hip.py | head -20"

echo
echo "===== do OTHER iris checkouts on this node expose hipIpcMemHandle_t? ====="
for d in /home/subvadla/amd-master/iris /home/subvadla/amd-master-ddt-*/iris; do
  h="$d/iris/hip.py"
  if [ -f "$h" ]; then
    n=$(grep -c 'hipIpcMemHandle_t' "$h" 2>/dev/null)
    g=$(grep -c 'def get_ipc_handle' "$h" 2>/dev/null)
    rev=$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo "no-git")
    echo "  $d rev=$rev hipIpcMemHandle_t=$n get_ipc_handle=$g"
  fi
done

echo
echo "===== how rank-1 builds its heap (the spawned helper) ====="
grep -n -A40 'CREATE_SHEMEM_CODE = ' "$R" | head -60

echo
echo "===== how the helper is launched ====="
grep -n 'CREATE_SHEMEM_CODE\|subprocess\|Popen\|sys.argv\[1\]' "$R" | head -20
