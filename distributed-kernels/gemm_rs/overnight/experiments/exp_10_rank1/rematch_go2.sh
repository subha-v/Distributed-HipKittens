#!/usr/bin/env bash
# Relaunch after the first attempt died: build.sh had been invoked on the HOST,
# where there is no hipcc, so all three modules "failed" and the launcher
# refused to measure. Two changes:
#   * verify the existing .so was not truncated by that failed build, then
#     measure it as-is. The rebuild was only ever insurance against a stale
#     binary, and shape 6 is a far cheaper and more direct test of the same
#     thing: pre-fix it read ~6400 us on generic_config, post-fix it must read
#     ~1970-2000 us. The sweep runs shape 5 (index) first, so we learn this in
#     the first few minutes and can still rebuild if it reads stale.
#   * write the launcher body here rather than relying on a pushed copy.
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1
SHAPES=${1:-5,3,0,1,2,4}
ITERS=${2:-12}
REPS=${3:-2}

if pgrep -f 'rematch_launch.sh|rematch_sweep.sh|mp_vs_rank1' > /dev/null; then
  echo "REFUSING: a rematch run is already alive"; pgrep -af 'rematch_'; exit 1
fi

echo "--- integrity of the module the harness will load ---"
ls -l --time-style=full-iso "$ON/harness/build/gemm_rs_mi300x.so" \
  "$ON/harness/build/dhk_rt.so" "$ON/harness/build/gemm_rs_mi300x_control.so"
SZ=$(stat -c%s "$ON/harness/build/gemm_rs_mi300x.so")
if [ "$SZ" != "373560" ]; then
  echo "ABORT: gemm_rs_mi300x.so is $SZ bytes, expected 373560 (pre-existing"
  echo "       gated build). The failed host-side build.sh may have damaged it;"
  echo "       rebuild inside dhk-gemmrs before measuring."
  exit 2
fi
echo "size 373560 unchanged -- the failed host build did not touch it"

cat > /tmp/rematch_launch2.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1
{
  echo "===== launcher2 start $(date -Is) ====="
  echo "--- clocks ---"
  rocm-smi --showperflevel 2>&1 | grep -iE 'Performance Level' | head -3
  rocm-smi --showclocks 2>&1 | grep -i 'sclk' | head -3
  echo "--- vs_logs must be empty (pre-fix set already archived) ---"
  ls "$E/vs_logs" 2>&1 | head -3
  echo "count: $(ls "$E/vs_logs" 2>/dev/null | wc -l)"
  echo "--- sweep ---"
  bash "$E/rematch_sweep.sh" "$1" "$2" "$3" 12700 1
  echo "===== launcher2 done $(date -Is) ====="
} > "$E/rematch_run.log" 2>&1
EOS

setsid nohup bash /tmp/rematch_launch2.sh "$SHAPES" "$ITERS" "$REPS" \
  < /dev/null > /dev/null 2>&1 &
echo "detached pid=$!  shapes=$SHAPES iters=$ITERS reps=$REPS"
sleep 5
head -12 "$E/rematch_run.log" 2>&1
echo "===== DONE ====="
