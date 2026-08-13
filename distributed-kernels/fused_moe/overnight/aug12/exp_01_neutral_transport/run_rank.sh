#!/usr/bin/env bash
# Guarded launcher for final two-rank Stage-0/1 transport points.
# It never changes branches or checkouts.  Every MPI launch is setsid+timeout.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BINARY=${BINARY:-"$SCRIPT_DIR/raw/rank-build/rank_transport"}
OUTPUT=${OUTPUT:-"$SCRIPT_DIR/raw/rank-points.jsonl"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$SCRIPT_DIR/raw/rank-artifacts"}
STAGE=0
PHASE=all
ITERATIONS=${ITERATIONS:-100}
POINT_TIMEOUT=${POINT_TIMEOUT:-1800}
GPU_IDS=${GPU_IDS:-0,1}
METHOD_CSV=${METHOD_CSV:-cu_push,cu_pull,host_copy_path}
LIFETIME_CSV=${LIFETIME_CSV:-one_epoch,ping_pong}

usage() {
  cat <<'EOF'
usage: bash run_rank.sh [--stage 0|1] [--gate|--soak|--run|--all] [options]
  --binary PATH
  --output PATH
  --artifact-dir PATH
  --methods CSV          default cu_push,cu_pull,host_copy_path
  --lifetimes CSV        Stage-0 default one_epoch,ping_pong
  --iterations N
  --timeout-seconds N
  --gpu-ids A,B          exactly two visible local GPUs

Timing is rejected by rank_transport unless exact gate and 600-epoch soak
artifacts already pass for the point.
EOF
}

while (($#)); do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --gate) PHASE=gate; shift ;;
    --soak) PHASE=soak; shift ;;
    --run) PHASE=run; shift ;;
    --all) PHASE=all; shift ;;
    --binary) BINARY=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --artifact-dir) ARTIFACT_DIR=$2; shift 2 ;;
    --methods) METHOD_CSV=$2; shift 2 ;;
    --lifetimes) LIFETIME_CSV=$2; shift 2 ;;
    --iterations) ITERATIONS=$2; shift 2 ;;
    --timeout-seconds) POINT_TIMEOUT=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "run_rank.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$STAGE" == 0 || "$STAGE" == 1 ]] || {
  echo "run_rank.sh: --stage must be 0 or 1" >&2
  exit 2
}
[[ "$PHASE" == gate || "$PHASE" == soak || "$PHASE" == run || "$PHASE" == all ]] || {
  echo "run_rank.sh: invalid phase" >&2
  exit 2
}
[[ -x "$BINARY" ]] || {
  echo "run_rank.sh: binary is missing or not executable: $BINARY" >&2
  exit 2
}
command -v mpirun >/dev/null || {
  echo '{"status":"blocked_mpi_toolchain","detail":"mpirun_not_found"}' >&2
  exit 42
}
command -v timeout >/dev/null || {
  echo "run_rank.sh: GNU timeout is required" >&2
  exit 2
}
command -v setsid >/dev/null || {
  echo "run_rank.sh: setsid is required" >&2
  exit 2
}

BRANCH=$(git -C "$REPO_ROOT" branch --show-current)
[[ "$BRANCH" == ablations ]] || {
  echo "run_rank.sh: branch must be ablations, found $BRANCH" >&2
  exit 9
}
LOCAL_HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD)
REMOTE_HEAD=$(git -C "$REPO_ROOT" ls-remote origin refs/heads/ablations |
  awk 'NR == 1 {print $1}')
[[ -n "$REMOTE_HEAD" ]] || {
  echo "run_rank.sh: cannot establish origin/ablations HEAD" >&2
  exit 9
}
[[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]] || {
  echo "run_rank.sh: local/remote HEAD mismatch local=$LOCAL_HEAD remote=$REMOTE_HEAD" >&2
  exit 9
}

IFS=',' read -r -a METHODS <<<"$METHOD_CSV"
IFS=',' read -r -a LIFETIMES <<<"$LIFETIME_CSV"
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
[[ ${#GPU_ARRAY[@]} == 2 ]] || {
  echo "run_rank.sh: --gpu-ids must name exactly two GPUs" >&2
  exit 2
}
for method in "${METHODS[@]}"; do
  [[ "$method" == cu_push || "$method" == cu_pull || "$method" == host_copy_path ]] || {
    echo "run_rank.sh: unsupported method $method" >&2
    exit 2
  }
done

mkdir -p "$ARTIFACT_DIR" "$(dirname -- "$OUTPUT")"
RUNNER_STATUS="$ARTIFACT_DIR/runner-status.jsonl"
PROVENANCE="$ARTIFACT_DIR/provenance.txt"

{
  echo "timestamp_utc=$(date -u +%FT%TZ)"
  echo "branch=$BRANCH"
  echo "local_head=$LOCAL_HEAD"
  echo "remote_head=$REMOTE_HEAD"
  echo "binary=$BINARY"
  echo "binary_sha256=$(sha256sum "$BINARY" | awk '{print $1}')"
  echo "gpu_ids=$GPU_IDS"
  echo "hostname=$(hostname -f 2>/dev/null || hostname)"
  echo "kernel=$(uname -srvmo)"
  echo "hipcc_version_begin"
  hipcc --version 2>&1 || true
  echo "hipcc_version_end"
  echo "mpirun_version_begin"
  mpirun --version 2>&1 || true
  echo "mpirun_version_end"
  echo "selected_environment_begin"
  env | LC_ALL=C sort | awk -F= '
    $1 ~ /^(HIP|HSA|ROCR|ROCM|OMPI|UCX)_/ ||
    $1 ~ /^(GPU_MAX_HW_QUEUES|LD_LIBRARY_PATH)$/ {
      print
    }'
  echo "selected_environment_end"
  echo "rocm_smi_begin"
  rocm-smi --showproductname --showserial --showclocks --showtemp --showpower \
    --showpids 2>&1 || true
  echo "rocm_smi_end"
  echo "numa_begin"
  numactl --hardware 2>&1 || true
  echo "numa_end"
} >"$PROVENANCE"

check_gpu_idle() {
  local contenders
  contenders=$(pgrep -af '(^|/)(mpirun|mpiexec|torchrun)( |$)|rank_transport|mok_synthetic' || true)
  if [[ -n "$contenders" ]]; then
    echo "run_rank.sh: GPU/process preflight found a possible contender:" >&2
    echo "$contenders" >&2
    return 1
  fi
  local smi_file="$ARTIFACT_DIR/gpu-pids-preflight.txt"
  rocm-smi --showpids >"$smi_file" 2>&1 || {
    echo "run_rank.sh: rocm-smi --showpids failed" >&2
    return 1
  }
  local active_rows
  active_rows=$(awk '
    $1 ~ /^[0-9]+$/ {
      # KFD table: PID, process, GPU(s), VRAM, SDMA, CU occupancy.
      # Zero-footprint gpuagent/UNKNOWN rows are not active GPU work.
      if (($4 + 0) > 0 || ($5 + 0) > 0 || ($6 + 0) > 0) print
    }
  ' "$smi_file")
  if [[ -n "$active_rows" ]]; then
    echo "run_rank.sh: rocm-smi reports active GPU work:" >&2
    echo "$active_rows" >&2
    return 1
  fi
}

MPI_BIND=()
if mpirun --version 2>&1 | awk 'BEGIN {ok=1} /Open MPI/ {ok=0} END {exit ok}'; then
  MPI_BIND=(--bind-to core --map-by ppr:1:numa --report-bindings)
fi

if [[ "$STAGE" == 0 ]]; then
  RECORD_SIZES=(65536)
else
  RECORD_SIZES=(256 1024 4096 14336 65536 262144 1048576 4194304 16777216 67108864)
  LIFETIMES=(ping_pong)
fi

run_phase() {
  local phase=$1 method=$2 size=$3 lifetime=$4 point_index=$5
  local phase_args=()
  case "$phase" in
    gate) phase_args=(--gate) ;;
    soak) phase_args=(--soak 600) ;;
    run) phase_args=(--run --iterations "$ITERATIONS") ;;
  esac
  check_gpu_idle
  local log="$ARTIFACT_DIR/${phase}-${method}-rb${size}-${lifetime}.log"
  local started
  started=$(date -u +%FT%TZ)
  set +e
  HIP_VISIBLE_DEVICES="$GPU_IDS" ROCR_VISIBLE_DEVICES="$GPU_IDS" \
    setsid timeout --signal=TERM "$POINT_TIMEOUT" \
      mpirun -np 2 "${MPI_BIND[@]}" \
      -x HIP_VISIBLE_DEVICES -x ROCR_VISIBLE_DEVICES \
      "$BINARY" "${phase_args[@]}" \
      --method "$method" --record-bytes "$size" \
      --total-bytes 67108864 --lifetime "$lifetime" \
      --artifact-dir "$ARTIFACT_DIR" --output "$OUTPUT" \
      >"$log" 2>&1
  local rc=$?
  set -e
  local classification=completed
  case "$rc" in
    0) classification=completed ;;
    124) classification=watchdog ;;
    3|4|5|6|7) classification=protocol_or_gate_failure ;;
    *) classification=launcher_or_runtime_failure ;;
  esac
  printf \
    '{"timestamp_utc":"%s","stage":%s,"phase":"%s","method":"%s","record_bytes":%s,"lifetime":"%s","rotation_index":%s,"exit_code":%s,"classification":"%s","log":"%s"}\n' \
    "$started" "$STAGE" "$phase" "$method" "$size" "$lifetime" \
    "$point_index" "$rc" "$classification" "$log" \
    >>"$RUNNER_STATUS"
  if ((rc != 0)); then
    echo "run_rank.sh: $classification rc=$rc log=$log" >&2
    return "$rc"
  fi
}

point_index=0
for size in "${RECORD_SIZES[@]}"; do
  for lifetime in "${LIFETIMES[@]}"; do
    # Rotate method order at every paired point.
    for ((offset = 0; offset < ${#METHODS[@]}; ++offset)); do
      method_index=$(((point_index + offset) % ${#METHODS[@]}))
      method=${METHODS[$method_index]}
      if [[ "$PHASE" == all ]]; then
        run_phase gate "$method" "$size" "$lifetime" "$point_index"
        run_phase soak "$method" "$size" "$lifetime" "$point_index"
        run_phase run "$method" "$size" "$lifetime" "$point_index"
      else
        run_phase "$PHASE" "$method" "$size" "$lifetime" "$point_index"
      fi
    done
    point_index=$((point_index + 1))
  done
done

echo "RANK_RUNNER_DONE stage=$STAGE phase=$PHASE output=$OUTPUT"
