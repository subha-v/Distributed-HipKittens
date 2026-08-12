#!/usr/bin/env bash
# ===========================================================================
# exp_22 -- ONE command that sweeps the saturation grid and emits
# saturation.json. Run it on the node (not in this repo) once the GPU lease is
# in hand:
#
#   bash run_saturation.sh --tier quick     # ~2 min smoke, 9 points
#   bash run_saturation.sh --tier plan      # the plan.md grid  (default)
#   bash run_saturation.sh --tier full      # + granularity / all-pushers /
#                                           #   fixed-work cross-checks
#
# Properties that matter if it dies halfway:
#   * the ubench appends ONE json object per point to saturation.jsonl and
#     fflushes, so a kill loses at most the point in flight;
#   * on restart it reads the keys already present and skips them, so this
#     script simply re-invokes it until the grid is empty (--attempts);
#   * saturation.json is regenerated from the jsonl every attempt, so a partial
#     sweep is still a usable (and clearly-marked partial) artifact.
#
# Node discipline: refuses to start while any other process holds a GPU, wraps
# the binary in setsid + timeout, and never SIGKILLs.
# ===========================================================================
set -uo pipefail

TIER=plan
OUTDIR=/home/subvadla/e22/out
ROTATIONS=5
WARMUP=2
ATTEMPTS=6
SLICE=900              # seconds per attempt; the binary stops cleanly at it
HARD=1200              # timeout(1) backstop per attempt
EXECPFX="docker exec subha_k1"
FORCE=0
EXTRA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER=$2; shift 2 ;;
    --out) OUTDIR=$2; shift 2 ;;
    --rotations) ROTATIONS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --attempts) ATTEMPTS=$2; shift 2 ;;
    --slice) SLICE=$2; shift 2 ;;
    --exec) EXECPFX=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --extra) EXTRA=$2; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

BIN=$OUTDIR/e22_saturation
JSONL=$OUTDIR/saturation_$TIER.jsonl
JSON=$OUTDIR/saturation.json
LOG=$OUTDIR/run_$TIER.log
mkdir -p "$OUTDIR"

if [ ! -x "$BIN" ]; then
  echo "FATAL: $BIN missing -- run build.sh first"; exit 2
fi

echo "=== exp_22 saturation sweep: tier=$TIER rotations=$ROTATIONS ==="
date -u +%FT%TZ

# ---- node lease guard --------------------------------------------------------
# rocm-smi --showpids is the check that actually works on this node. It may not
# be on the host PATH, in which case fall back to the container -- and if NEITHER
# works, say so loudly instead of silently "passing" the guard.
echo "--- rocm-smi --showpids (must list no processes; --force overrides) ---"
SMI=""
if command -v rocm-smi >/dev/null 2>&1; then
  SMI="rocm-smi"
elif $EXECPFX rocm-smi --version >/dev/null 2>&1; then
  SMI="$EXECPFX rocm-smi"
fi
if [ -z "$SMI" ]; then
  echo "WARNING: rocm-smi not found on the host or in the container; the"
  echo "         one-GPU-job-at-a-time guard could NOT be evaluated."
  if [ "$FORCE" -eq 0 ]; then
    echo "FATAL: refusing to run blind. Re-run with --force only after checking"
    echo "       by hand that no other job holds the GPUs."
    exit 3
  fi
else
  $SMI --showpids > "$OUTDIR/showpids_before.txt" 2>&1
  cat "$OUTDIR/showpids_before.txt"
  # gpuagent is the always-resident management daemon, not a job; anything else
  # holding a GPU is.
  if grep -E '^[0-9]+[[:space:]]' "$OUTDIR/showpids_before.txt" \
      | grep -qv gpuagent; then
    echo "another process holds a GPU. One 8-GPU job at a time."
    if [ "$FORCE" -eq 0 ]; then
      echo "FATAL: refusing to start. Re-run with --force only after confirming"
      echo "       the lease is ours."
      exit 3
    fi
  fi
fi

# ---- provenance (clocks are sampled per attempt, not per point: an amd-smi
# ---- call costs ~1 s, which would dominate a 30 ms measurement. Drift shows up
# ---- in the per-point rotation spread, which is recorded for every point.)
{
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "tier=$TIER rotations=$ROTATIONS warmup=$WARMUP"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "src_sha256=$(sha256sum /home/subvadla/e22/src/e22_saturation.hip | cut -d' ' -f1)"
  echo "src_rev=$(grep -m1 'define E22_SRC_REV' /home/subvadla/e22/src/e22_saturation.hip)"
  echo "dhk_head=$(git -C /home/subvadla/e22/DHK rev-parse HEAD 2>/dev/null)"
  echo "--- clocks ---"
  rocm-smi --showclocks 2>&1 | head -40
} > "$OUTDIR/env_before.txt" 2>&1
sed -n '1,12p' "$OUTDIR/env_before.txt"

# ---- the sweep ---------------------------------------------------------------
: > "$LOG"
for a in $(seq 1 "$ATTEMPTS"); do
  BEFORE=$( [ -f "$JSONL" ] && wc -l < "$JSONL" || echo 0 )
  echo "--- attempt $a/$ATTEMPTS (points so far: $BEFORE) ---" | tee -a "$LOG"
  setsid timeout "$HARD" $EXECPFX "$BIN" \
      --tier "$TIER" --out "$JSONL" \
      --rotations "$ROTATIONS" --warmup "$WARMUP" \
      --max-seconds "$SLICE" $EXTRA >> "$LOG" 2>&1
  RC=$?
  # timeout(1) kills `docker exec`, NOT the process inside the container, so a
  # fired backstop would leave an orphan holding all 8 GPUs. The binary's own
  # --max-seconds is set below the backstop precisely so this path is not taken,
  # but if it is: SIGTERM inside the container, graceful, never SIGKILL.
  if [ "$RC" -eq 124 ]; then
    echo "backstop timeout fired; SIGTERM-ing the in-container binary" | tee -a "$LOG"
    $EXECPFX pkill -TERM -f e22_saturation >> "$LOG" 2>&1
    sleep 10
    $EXECPFX pgrep -af e22_saturation | tee -a "$LOG"
  fi
  AFTER=$( [ -f "$JSONL" ] && wc -l < "$JSONL" || echo 0 )
  echo "attempt $a rc=$RC points $BEFORE -> $AFTER" | tee -a "$LOG"
  # regenerate the artifact every attempt so a partial sweep is still usable
  $EXECPFX python3 /home/subvadla/e22/src/summarize.py "$JSONL" "$JSON" \
      >> "$LOG" 2>&1
  if [ "$AFTER" -le "$BEFORE" ] && [ "$RC" -eq 0 ]; then
    echo "no new points and clean exit: grid complete"; break
  fi
  if [ "$AFTER" -le "$BEFORE" ] && [ "$RC" -ne 0 ]; then
    echo "WARNING: attempt made no progress and failed (rc=$RC); see $LOG"
    tail -20 "$LOG"
    break
  fi
done

{
  echo "utc=$(date -u +%FT%TZ)"
  rocm-smi --showclocks 2>&1 | head -40
} > "$OUTDIR/env_after.txt" 2>&1

echo "=== summary ==="
$EXECPFX python3 /home/subvadla/e22/src/summarize.py "$JSONL" "$JSON" --print
echo "artifacts: $JSONL  $JSON  $LOG"
date -u +%FT%TZ
