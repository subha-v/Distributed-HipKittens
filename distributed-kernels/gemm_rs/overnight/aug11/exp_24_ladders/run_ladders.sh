#!/usr/bin/env bash
# exp_24 -- the external ladders orchestrator.
#
#   ours  vs  reference GEMM+RCCL  vs  frozen rank-1,
#   same node, same run, both protocols, all six graded shapes.
#
# Two instruments, for the reasons in design.md 1:
#   A  ladder_mp.py  -- five arms in ONE 8-process pool, graded AND pipelined,
#                       full rotation, raw samples. This is the ladder.
#   B  the official evaluator, via the EXISTING drivers, arm order rotated.
#      This is the cross-check; eval.py cannot produce medians, samples, a fixed
#      iteration count, a pipelined region, or a null arm.
#
# Nothing under tools/, harness/ or experiments/ is modified. The frozen rank-1
# submission is never opened for writing; tools/patch_rank1.py enforces its hash
# on every run and refuses to emit on a mismatch.
#
# Usage:
#   bash run_ladders.sh [phase]      phase in: all preflight stage A B parse
# Env:
#   LAD_QUICK=1       one shape, reduced reps, skip instrument B  (~6 min)
#   LAD_SHAPES=csv    shape indices for instrument A (default rotated 0..5)
#   LAD_ITERS=50      graded iterations per arm per rep
#   LAD_REPS=5        reps == arm count, so every arm is first exactly once
#   LAD_BURST=10      back-to-back calls per pipelined sample
#   LAD_PIPE_ITERS=20 pipelined samples per arm per rep
#   LAD_ROTATIONS=2   instrument B arm-order rotations
#   LAD_ALLOW_DIRTY=1 proceed even if another job holds the GPUs (NOT for timing)
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
RAW=$D/raw
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
IRISDST=/usr/local/lib/python3.10/dist-packages
CB=$ON/compbench

PHASE=${1:-all}
QUICK=${LAD_QUICK:-0}
ITERS=${LAD_ITERS:-50}
REPS=${LAD_REPS:-5}
BURST=${LAD_BURST:-10}
PIPE_ITERS=${LAD_PIPE_ITERS:-20}
ROTATIONS=${LAD_ROTATIONS:-2}
PORT0=${LAD_PORT0:-13100}

# A quick run's samples must never be reachable by a full ladder's aggregation.
LADDIR=ladder
if [ "$QUICK" = "1" ]; then
  SHAPES=${LAD_SHAPES:-1}
  ITERS=${LAD_ITERS:-8}; REPS=${LAD_REPS:-2}; PIPE_ITERS=${LAD_PIPE_ITERS:-4}
  ROTATIONS=0; LADDIR=ladder_quick
else
  # Rotated so no shape is systematically first; 5 and 6 lead because they are
  # the two rows that carry the graded gap and the ones worth having if the
  # night is cut short.
  SHAPES=${LAD_SHAPES:-4,5,0,2,1,3}
fi

mkdir -p "$RAW/eval" "$RAW/$LADDIR" "$D/logs"

say() { echo "[$(date -Is)] $*"; }
hr()  { echo "----------------------------------------------------------------"; }

# ---------------------------------------------------------------- preflight ---
# rocm-smi --showpids for all three arms: run_reference_arm.sh does this, the
# other two drivers do NOT. One 8-GPU job of ours at a time.
kfd_fds() { ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd; }
kfd_pids() { rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/{print $1}'; }

preflight() {
  say "=== preflight ==="
  local pids; pids=$(kfd_pids)
  echo "rocm-smi --showpids:"; rocm-smi --showpids 2>&1 | sed -n '1,20p'
  echo "kfd fds: $(kfd_fds)"
  if [ -n "$pids" ]; then
    say "NODE DIRTY: KFD pids [$pids] hold the GPUs"
    # LAD_LEASED=1 means tools/gpu_lease.sh already took the lock AND performed
    # its own wait-for-drain, so this check is a report rather than a gate --
    # duplicating the abort here would only add a way to lose an acquired lease
    # to a worker that is still exiting.
    if [ "${LAD_LEASED:-0}" = "1" ]; then
      say "LAD_LEASED=1 -- the lease already drained the node; continuing"
    elif [ "${LAD_ALLOW_DIRTY:-0}" != "1" ]; then
      say "ABORT. Another job owns the devices; a ladder measured against it is void."
      return 1
    else
      say "LAD_ALLOW_DIRTY=1 -- proceeding, and this run MUST NOT be reported as timing"
    fi
  fi
  # Bounded drain wait. SIGTERM only, never SIGKILL: this kernel uses HIP IPC and
  # leaked mappings wedge the node.
  for _ in $(seq 1 24); do
    [ "$(kfd_fds)" = "0" ] && break
    say "  waiting for kfd drain (fds=$(kfd_fds))"; sleep 5
  done

  say "--- pin clocks (idle sclk is ~125 MHz; unpinned short runs are unrepeatable) ---"
  bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -6 || say "WARN: clock pin failed"
  rocm-smi --showsclkrange 2>&1 | sed -n '1,6p'
  return 0
}

# --------------------------------------------------------------- provenance ---
provenance() {
  say "=== provenance -> $D/logs/provenance.txt ==="
  {
    echo "host      : $(hostname)"
    echo "date      : $(date -Is)"
    # /home/subvadla/dhk is NOT a git worktree -- it is an scp-synced copy made by
    # tools/push_scoped.ps1. An empty `git HEAD:` line here would read as a failed
    # command rather than as an absent repo, so say which it is; the module
    # sha256s below are the actual provenance for what was measured.
    if git -C /home/subvadla/dhk rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "git HEAD  : $(git -C /home/subvadla/dhk rev-parse HEAD)"
      echo "git branch: $(git -C /home/subvadla/dhk rev-parse --abbrev-ref HEAD)"
    else
      echo "git       : node copy is NOT a git worktree (scp-synced from the"
      echo "            Windows worktree on branch GEMM-RS); provenance for what"
      echo "            was measured is the module sha256 set below"
    fi
    echo
    echo "--- module fingerprints (the config under test) ---"
    for f in gemm_rs_mi300x.so dhk_rt.so; do
      if [ -f "$ON/harness/build/$f" ]; then
        echo "$f sha256 $(sha256sum "$ON/harness/build/$f" | cut -c1-32) \
mtime $(date -Is -r "$ON/harness/build/$f")"
      else
        echo "$f MISSING"
      fi
    done
    echo -n "submission.py == hk_submission.py: "
    cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" \
      && echo yes || echo NO
    echo
    echo "--- scored_shapes (BM/BN/BK/NR/config_row) ---"
    sed -n '/inline constexpr std::array<shape_entry, 6> scored_shapes/,/}};/p' \
      "$ON/../gemm_rs_mi300x_host_abi.hpp"
    echo
    echo "--- frozen rank-1 hash gate ---"
    sha256sum "$RANK1"
    echo "expected 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5"
    echo
    echo "--- clocks ---"
    rocm-smi --showsclkrange 2>&1 | sed -n '1,10p'
  } > "$D/logs/provenance.txt" 2>&1
  cat "$D/logs/provenance.txt"
}

# -------------------------------------------------------------------- stage ---
# Instrument A needs all three arms staged WITHOUT an evaluator run, so staging
# lives here rather than being a side effect of the tools/ drivers. These are the
# drivers' own cp lines, plus tools/patch_rank1.py at its real path.
stage_arms() {
  say "=== stage arms ==="
  local common="eval.py task.py utils.py reference.py"
  for arm in ours reference rank1; do
    mkdir -p "$CB/$arm/.tmp"
    for f in $common; do cp "$SRC/$f" "$CB/$arm/$f"; done
    cp "$SRC/cases.txt" "$CB/$arm/cases_test.txt"
    cat > "$CB/$arm/cases_bench.txt" <<'EOF'
world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42
EOF
  done

  cp "$ON/harness/hk_submission.py" "$CB/ours/submission.py"
  cp "$SRC/submission.py"           "$CB/reference/submission.py"

  # Hash-gated: patch_rank1.py refuses to emit unless the frozen source hashes
  # to 7940fcb8...f0dc5. The frozen file is never opened for writing.
  python3 "$ON/tools/patch_rank1.py" "$RANK1" "$CB/rank1/submission.py" \
    | tee "$D/logs/patch_rank1.txt"
  if ! grep -q '^wrote ' "$D/logs/patch_rank1.txt"; then
    say "ABORT: rank-1 hash gate refused. The frozen submission changed."
    return 1
  fi

  # Stale IPC handles and a stale Triton cache both make the buffer-ops knob look
  # ineffective (the cached hsaco has buffer_store baked in).
  rm -f  "$CB/rank1"/ipc_handles_rank*.bin "$CB/rank1"/*.pkl
  rm -rf "$CB/rank1/.triton"

  echo "--- reference arm is what? (first 12 lines) ---"
  sed -n '1,12p' "$CB/reference/submission.py"
  for arm in ours reference rank1; do
    echo "staged $arm: $(ls "$CB/$arm" | tr '\n' ' ')"
  done
  return 0
}

# ------------------------------------------- instrument A: the real ladder ---
# All five arms in one pool per shape. Container env carries the two disclosed
# rank-1 repairs (buffer-ops knob + compat sitecustomize) and the no-op sudo.
instrument_a() {
  say "=== instrument A: same-run interleaved ladder, both protocols ==="
  local out="$RAW/$LADDIR"
  mkdir -p "$out"
  IFS=',' read -ra LIST <<< "$SHAPES"
  for s in "${LIST[@]}"; do
    local port=$((PORT0 + s)) t0; t0=$(date +%s)
    hr; say "shape index $s"
    rm -f "$CB/rank1"/ipc_handles_rank*.bin
    for _ in $(seq 1 24); do
      [ "$(kfd_fds)" = "0" ] && break
      say "  waiting for kfd drain (fds=$(kfd_fds))"; sleep 5
    done
    docker exec -w "$CB/rank1" \
      -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
      -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
      -e PYTHONUNBUFFERED=1 \
      -e HK_DEBUG=0 \
      -e HK_BUILD_DIR="$ON/harness/build" \
      -e TRITON_CACHE_DIR="$CB/rank1/.triton" \
      -e TORCH_EXTENSIONS_DIR="$CB/rank1/.ext" \
      -e TMPDIR="$CB/rank1/.tmp" \
      -e HSA_ENABLE_COREDUMP=0 \
      -e AMDGCN_USE_BUFFER_OPS=0 \
      -e VS_FORCE_BIAS=1 \
      -e LAD_OUT="$out/lad_s${s}" \
      -e LAD_ARMS="${LAD_ARMS:-}" \
      -e LAD_WARM_MS="${LAD_WARM_MS:-400}" \
      dhk-gemmrs bash -lc \
        "timeout --signal=TERM ${LAD_TMO:-1800} setsid python3 -u $D/ladder_mp.py \
         $s $ITERS $REPS $port $BURST $PIPE_ITERS" \
      2>&1 | tee "$D/logs/ladder_s${s}.log" \
      | grep -vE '^\[1/|^\[2/|^\[3/|hipcc|^ *[0-9]+ \||warning:|^ *\^|preprocessed|replaced kernel|unsupported CUDA'
    say "shape $s wall=$(( $(date +%s) - t0 ))s"
    grep -c SHIM_WAS_CALLED "$out/lad_s${s}".rank*.stderr 2>/dev/null \
      | awk -F: '$2>0 {print "SHIM_WAS_CALLED in", $1, "-- repair #3 is no longer dead code"}'
  done
}

# ------------------------------ instrument B: the official evaluator, rotated ---
# eval.py hardcodes MASTER_PORT=12356, so these are strictly serialized.
# Every driver starts with `rm -rf $DIR`, so each arm's raw output is copied out
# before the next arm runs.
capture() {  # capture <arm> <dest>
  local arm="$1" dest="$2"
  mkdir -p "$dest"
  cp "$CB/$arm"/*.popcorn.txt "$dest/" 2>/dev/null
  cp "$CB/$arm"/*.stdout.txt  "$dest/" 2>/dev/null
  cp "$CB/$arm"/*.stderr.txt  "$dest/" 2>/dev/null
  echo "captured $(ls "$dest" | tr '\n' ' ')"
}

drain() {
  for _ in $(seq 1 36); do
    [ "$(kfd_fds)" = "0" ] && break
    say "  waiting for kfd drain (fds=$(kfd_fds))"; sleep 5
  done
  sleep 20   # let eval.py's hardcoded MASTER_PORT 12356 leave TIME_WAIT
}

eval_arm() {  # eval_arm <arm> <rotation>
  local arm="$1" rot="$2" dest="$RAW/eval/rot${rot}/${arm}"
  hr; say "instrument B rot=$rot arm=$arm"
  echo "preflight kfd pids: [$(kfd_pids)]  fds: $(kfd_fds)"
  drain
  case "$arm" in
    ours)
      bash "$ON/tools/run_ours_evaluator.sh" 2>&1 \
        | tee "$D/logs/eval_rot${rot}_ours.log" | tail -40
      ;;
    reference)
      bash "$ON/tools/run_reference_arm.sh" 2>&1 \
        | tee "$D/logs/eval_rot${rot}_reference.log" | tail -40
      ;;
    rank1)
      # tools/run_rank1_bench3.sh was REPAIRED 2026-08-12 and is now the driver:
      # it passes AMDGCN_USE_BUFFER_OPS=0 via `docker exec -e`, points PYTHONPATH
      # at the real $ON/tools/compat, calls $ON/tools/patch_rank1.py, and runs in
      # dhk-gemmrs as uid 15523 instead of the root container. It performs all
      # three passes itself in the mandatory warm -> test -> bench order and has
      # its own node-clean preflight. The repair changes only how the arm is
      # LAUNCHED, never what it computes; disclosed in result.md 8.8.
      bash "$ON/tools/run_rank1_bench3.sh" 2>&1 \
        | tee "$D/logs/eval_rot${rot}_rank1.log" | tail -60
      # bench3 writes {warm,test,bench}.{popcorn,stdout,stderr}.txt, so one
      # capture takes all three passes.
      capture rank1 "$dest"
      return 0
      ;;
  esac
  capture "$arm" "$dest"
}

instrument_b() {
  if [ "$ROTATIONS" = "0" ]; then say "=== instrument B skipped ==="; return 0; fi
  say "=== instrument B: official evaluator, arm order rotated ==="
  local arms=(ours reference rank1) n=3
  for rot in $(seq 0 $((ROTATIONS - 1))); do
    for i in 0 1 2; do
      eval_arm "${arms[$(( (rot + i) % n ))]}" "$rot"
    done
  done
}

# -------------------------------------------------------------------- parse ---
# Aggregation runs TWICE: once the moment instrument A finishes, so the ladder --
# which is the deliverable -- is on disk and validated before instrument B spends
# hours in eval.py, and again after B. Otherwise a B that runs long or dies takes
# a completed 6-shape ladder down with it at the --expect-b assertion.
parse() {  # parse <expect_b> [fatal|soft]
  local expect_b="$1" mode="${2:-fatal}"
  local n_shapes; n_shapes=$(awk -F, '{print NF}' <<< "$SHAPES")
  local out="$D/ladders.json"
  [ "$QUICK" = "1" ] && out="$D/ladders_quick.json"
  say "=== aggregate -> $(basename "$out")  (expect-a=$n_shapes expect-b=$expect_b) ==="
  python3 "$D/ladders.py" --root "$D" --out "$out" --ladder-dir "$LADDIR" \
    --expect-a "$n_shapes" --expect-b "$expect_b"
  local rc=$?
  if [ $rc -ne 0 ] && [ "$mode" = "soft" ]; then
    say "WARN: strict aggregation failed (rc=$rc). Re-running with expect-b=0 so"
    say "      the instrument-A ladder is preserved; NOTE THIS IN result.md."
    python3 "$D/ladders.py" --root "$D" --out "$out" --ladder-dir "$LADDIR" \
      --expect-a "$n_shapes" --expect-b 0
    rc=$?
  fi
  return $rc
}

# ---------------------------------------------------------------------- main ---
say "exp_24 run_ladders.sh phase=$PHASE quick=$QUICK shapes=$SHAPES \
iters=$ITERS reps=$REPS pipe=${PIPE_ITERS}x${BURST} rotations=$ROTATIONS"

rc=0
case "$PHASE" in
  preflight) preflight; rc=$? ;;
  stage)     stage_arms; rc=$? ;;
  A)         preflight && stage_arms && instrument_a && parse 0; rc=$? ;;
  B)         preflight && stage_arms && instrument_b && parse "$ROTATIONS" soft; rc=$? ;;
  parse)     parse "$ROTATIONS" soft; rc=$? ;;
  all)
    preflight || exit 1
    provenance
    stage_arms || exit 1
    instrument_a
    parse 0 || say "WARN: instrument-A aggregation failed -- see the assertion above"
    cp -f "$D/ladders.json" "$D/ladders_instrumentA.json" 2>/dev/null
    instrument_b
    parse "$ROTATIONS" soft; rc=$?
    ;;
  *) say "unknown phase: $PHASE"; rc=2 ;;
esac

say "=== run_ladders.sh DONE phase=$PHASE rc=$rc ==="
exit $rc
