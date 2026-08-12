#!/usr/bin/env bash
# Benchmark the frozen rank-1 MI300X submission with the official evaluator.
#
# The submission itself is NEVER edited: it is hash-frozen (sha256 7940fcb8..
# ..f0dc5) and this script patches a COPY via tools/patch_rank1.py, which
# refuses to run if the source hash does not match.
#
# The four disclosed compatibility repairs, all argued behaviour-preserving:
#   1. iris staged at the literal python3.10 path the submission reads, and on
#      PYTHONPATH. (Verified importable -- `iris` and `iris.hip` -- in both
#      containers.)
#   2. a no-op `sudo` shim first on PATH, so the submission's sed against
#      iris/__init__.py cannot fire.
#   3. compat/sitecustomize.py supplies triton's wrap_handle_tensor_descriptor
#      as a stub that RAISES if called -- a dead branch for these kernels, so a
#      call would be a disclosure-worthy behaviour change and is made loud.
#   4. patch_rank1.py fixes the packed_metadata field count (6 -> 3), which is
#      behaviour-preserving because the dropped fields are never read.
#
# AMDGCN_USE_BUFFER_OPS=0 is REQUIRED, not optional: Triton 3.6.0 otherwise
# lowers rank-1's peer stores to buffer_store_dwordx2, whose voffset is 32-bit,
# which truncates a -4.4-billion-element offset and faults. The Triton cache is
# wiped on every run because that knob changes codegen and a warm cache would
# silently serve the faulting kernel.
#
# Pass order matters and must not be "simplified": the first pass warms the
# Triton JIT cache, because eval.py's TEST mode hardcodes a 60 s per-rank
# timeout that a cold compile of this kernel exceeds.
#
# ---------------------------------------------------------------------------
# REPAIRED 2026-08-12 (aug11/exp_24). As written before this date the script
# could not run rank-1 at all, and it predated exp_10's repair #6. Four defects,
# each independently fatal, all found by dry-run inspection rather than by a
# failed run -- which is how a "runs but measures nothing" tool survives:
#   a. AMDGCN_USE_BUFFER_OPS=0 was absent, so the arm faults. Worse, the env was
#      built as a single ENVS string interpolated into `bash -c`, so the knob
#      could not be injected from outside either. Now passed as `docker exec -e`
#      flags, which are both visible and overridable.
#   b. PYTHONPATH pointed at $ON/compat, which does not exist -- the tree is
#      $ON/tools/compat -- so repair #3's sitecustomize was silently absent.
#   c. It invoked $ON/patch_rank1.py, which does not exist; the tool is at
#      $ON/tools/patch_rank1.py. (The `if [ $? -ne 0 ]` guard below the call
#      also tested the wrong command's status, since `run` had already returned.)
#   d. It ran in container dhk-eval, whose /usr/local/shim holds the sudo shim
#      but which runs as ROOT, so every artifact it writes under the repo comes
#      out root-owned and breaks later sed/scp steps. dhk-gemmrs runs as uid
#      15523; its own `sudo` fails with "you do not exist in the passwd
#      database", which is exactly why the PATH shim is needed there.
# The env/container arrangement below now mirrors experiments/exp_10_rank1/
# r1_eval.sh, the driver that actually produced exp_10's measured comparison.
# ---------------------------------------------------------------------------
set -uo pipefail

NAME=dhk-gemmrs
IRISDST=/usr/local/lib/python3.10/dist-packages
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
DIR=$ON/compbench/rank1
COMPAT=$ON/tools/compat
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
TMO=${TMO:-1700}

run() { docker exec "$NAME" bash -c "$1"; }

# Node-clean preflight. run_reference_arm.sh did this and the other two arms did
# not, which meant a rank-1 number could be taken against a node still draining
# someone else's job. Wait rather than abort on the first sample: a previous
# run's last worker can linger for tens of seconds after its parent returns.
echo "===== node-clean preflight ====="
for i in $(seq 1 30); do
  n=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)
  if [ "$n" = "0" ]; then echo "node clean after $((i*10-10))s"; break; fi
  [ "$i" = "1" ] && echo "KFD pids: $n -- waiting for the node to drain"
  if [ "$i" = "30" ]; then
    echo "ABORT: node still dirty after 300s"
    rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
    exit 1
  fi
  sleep 10
done

echo "===== stage arm + apply the metadata repair ====="
run "rm -rf $DIR && mkdir -p $DIR/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $DIR/ && cp $SRC/cases.txt $DIR/cases_test.txt"
run "python3 $ON/tools/patch_rank1.py $RANK1 $DIR/submission.py" || { echo "patch failed (hash gate or path) - stopping"; exit 1; }

run "cat > $DIR/cases_bench.txt <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
echo staged"

go() {
  local mode="$1" cases="$2" label="$3"
  echo
  echo "################################################################"
  echo "# rank1 [$label] : eval.py $mode $cases (buffer ops disabled)"
  echo "################################################################"
  # The Triton cache is per-pass-wiped only for the warm pass; the later passes
  # must REUSE it, since warming it is the entire reason the warm pass exists.
  [ "$label" = "warm" ] && run "rm -rf $DIR/.triton"
  run "rm -f $DIR/ipc_handles_rank*.bin $DIR/*.pkl"
  docker exec -w "$DIR" \
    -e PATH="$COMPAT/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
    -e PYTHONPATH="$COMPAT:$IRISDST" \
    -e PYTHONUNBUFFERED=1 \
    -e POPCORN_FD=3 -e POPCORN_GPUS=8 \
    -e TRITON_CACHE_DIR="$DIR/.triton" \
    -e TORCHINDUCTOR_CACHE_DIR="$DIR/.inductor" \
    -e TORCH_EXTENSIONS_DIR="$DIR/.ext" \
    -e TMPDIR="$DIR/.tmp" \
    -e HSA_ENABLE_COREDUMP=0 \
    -e AMDGCN_USE_BUFFER_OPS=0 \
    "$NAME" bash -lc "timeout --signal=TERM $TMO setsid python3 -u eval.py $mode $cases \
       3>$DIR/$label.popcorn.txt >$DIR/$label.stdout.txt 2>$DIR/$label.stderr.txt"
  echo "exit=$?"
  echo "--- popcorn ---"
  run "cat $DIR/$label.popcorn.txt 2>/dev/null | head -90"
  echo "--- shim invoked? (must print 0) ---"
  run "grep -c SHIM_WAS_CALLED $DIR/$label.stderr.txt 2>/dev/null || echo 0"
  echo "--- faults / errors ---"
  run "grep -E 'Memory access fault|Traceback|Error|error:' $DIR/$label.stderr.txt 2>/dev/null | sort | uniq -c | sort -rn | head -10"
  echo "--- stderr tail ---"
  run "tail -14 $DIR/$label.stderr.txt 2>/dev/null"
}

go benchmark cases_bench.txt warm
go test cases_test.txt test
go benchmark cases_bench.txt bench

echo
echo "===== DONE ====="
