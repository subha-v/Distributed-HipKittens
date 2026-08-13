#!/usr/bin/env bash
# Guarded, resumable launcher for the one-process/two-GPU diagnostic.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-"$HERE/build"}
BIN=${BIN:-"$OUT_DIR/neutral_transport"}
JSONL=${JSONL:-"$OUT_DIR/transport_diagnostic.jsonl"}
SUMMARY=${SUMMARY:-"$OUT_DIR/transport_crossover.json"}
LOG=${LOG:-"$OUT_DIR/run_diagnostic.log"}
HARD_TIMEOUT=7200
FORCE=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hard-timeout)
      HARD_TIMEOUT=$2
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$OUT_DIR"
if [[ ! -x "$BIN" ]]; then
  echo "FATAL: missing executable $BIN; run build.sh first" >&2
  exit 2
fi

SMI=()
if command -v rocm-smi >/dev/null 2>&1; then
  SMI=(rocm-smi)
elif command -v amd-smi >/dev/null 2>&1; then
  SMI=(amd-smi)
else
  echo "FATAL: neither rocm-smi nor amd-smi is available; refusing blind launch" >&2
  exit 3
fi

echo "=== GPU process guard ==="
if [[ ${SMI[0]} == rocm-smi ]]; then
  "${SMI[@]}" --showpids >"$OUT_DIR/showpids_before.txt" 2>&1
else
  "${SMI[@]}" process --json >"$OUT_DIR/showpids_before.txt" 2>&1
fi
awk '{print}' "$OUT_DIR/showpids_before.txt"

# rocm-smi process rows begin with a numeric PID on the target node. gpuagent
# is the resident management process and is not a benchmark occupant. Inside
# the compile container it can be rendered as UNKNOWN with zero VRAM, SDMA,
# and CU occupancy, which is likewise not a benchmark occupant.
if awk '
    /^[[:space:]]*[0-9]+[[:space:]]/ {
      if ($2 == "gpuagent") next
      if ($2 == "UNKNOWN" && $4 == 0 && $5 == 0 && $6 == 0) next
      found=1
    }
    END {exit found ? 0 : 1}
  ' "$OUT_DIR/showpids_before.txt"; then
  if [[ $FORCE -ne 1 ]]; then
    echo "FATAL: another process holds a GPU; refusing to overlap jobs" >&2
    exit 3
  fi
  echo "WARNING: --force bypassed a non-empty GPU process guard"
fi

{
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "branch=$(git -C "$HERE" branch --show-current 2>/dev/null || true)"
  echo "head=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || true)"
  echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "source_sha256=$(sha256sum "$HERE/neutral_transport.hip" | awk '{print $1}')"
  awk '/define NEUTRAL_TRANSPORT_SRC_REV/{print "source_rev="$3; exit}' \
    "$HERE/neutral_transport.hip"
} >"$OUT_DIR/provenance.txt"

echo "=== diagnostic ==="
echo "output=$JSONL timeout=${HARD_TIMEOUT}s"
set +e
setsid timeout --signal=TERM "$HARD_TIMEOUT" \
  "$BIN" --out "$JSONL" "${ARGS[@]}" >>"$LOG" 2>&1
rc=$?
set -e
if [[ $rc -eq 124 ]]; then
  echo "FATAL: diagnostic timed out and received SIGTERM; no retry was attempted" >&2
fi

python3 "$HERE/summarize.py" "$JSONL" "$SUMMARY" --print
summary_rc=$?
echo "artifacts: $JSONL $SUMMARY $LOG"
if [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
exit "$summary_rc"
