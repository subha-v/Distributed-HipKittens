#!/usr/bin/env bash
# CPU-only build for the two-rank HIP+MPI transport executable.
# No command in this script launches a GPU kernel.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DHK_ROOT=${DHK_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)}
OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/raw/rank-build"}
SOURCE=${SOURCE:-"$SCRIPT_DIR/rank_transport.hip"}
WANT_ISA=0

for arg in "$@"; do
  case "$arg" in
    --isa|isa) WANT_ISA=1 ;;
    --help|-h)
      echo "usage: bash build_rank.sh [--isa]"
      echo "env: DHK_ROOT, OUT_DIR, SOURCE, HIPCC, MPICXX"
      exit 0
      ;;
    *)
      echo "build_rank.sh: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"
HIPCC=${HIPCC:-$(command -v hipcc 2>/dev/null || true)}
MPICXX=${MPICXX:-$(command -v mpicxx 2>/dev/null || true)}

blocked() {
  local detail=$1
  printf '{"status":"blocked_mpi_toolchain","detail":"%s","source":"%s"}\n' \
    "$detail" "$SOURCE" | tee "$OUT_DIR/build-status.json"
  exit 42
}

[[ -n "$HIPCC" ]] || blocked "hipcc_not_found"
[[ -n "$MPICXX" ]] || blocked "mpicxx_not_found_in_remote_container"
[[ -f "$SOURCE" ]] || {
  echo "build_rank.sh: source not found: $SOURCE" >&2
  exit 2
}
[[ -d "$DHK_ROOT/include" ]] || {
  echo "build_rank.sh: invalid DHK_ROOT (missing include): $DHK_ROOT" >&2
  exit 2
}

MPI_COMPILE_TEXT=$("$MPICXX" --showme:compile 2>/dev/null || true)
MPI_LINK_TEXT=$("$MPICXX" --showme:link 2>/dev/null || true)
if [[ -z "$MPI_COMPILE_TEXT" || -z "$MPI_LINK_TEXT" ]]; then
  MPI_SHOW=$("$MPICXX" -show 2>/dev/null || true)
  [[ -n "$MPI_SHOW" ]] || blocked "mpicxx_present_but_flags_unavailable"
  read -r -a MPI_SHOW_ARGS <<<"$MPI_SHOW"
  ((${#MPI_SHOW_ARGS[@]} >= 2)) || blocked "mpicxx_show_output_unparseable"
  MPI_COMPILE=()
  MPI_LINK=()
  for token in "${MPI_SHOW_ARGS[@]:1}"; do
    case "$token" in
      -I*|-D*|-pthread) MPI_COMPILE+=("$token") ;;
      -L*|-l*|-Wl,*|-pthread) MPI_LINK+=("$token") ;;
    esac
  done
else
  read -r -a MPI_COMPILE <<<"$MPI_COMPILE_TEXT"
  read -r -a MPI_LINK <<<"$MPI_LINK_TEXT"
fi

FLAGS=(
  --offload-arch=gfx950
  -std=c++20
  -O3
  -DKITTENS_CDNA4
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -Rpass-analysis=kernel-resource-usage
  -I"$DHK_ROOT/include"
)

{
  echo "status=building"
  echo "source=$SOURCE"
  echo "source_sha256=$(sha256sum "$SOURCE" | awk '{print $1}')"
  echo "source_rev=$(awk '/^#define EXP01_RANK_SRC_REV / {print $3; exit}' "$SOURCE")"
  echo "dhk_head=$(git -C "$DHK_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "hipcc=$HIPCC"
  "$HIPCC" --version | awk 'NR <= 4'
  echo "mpicxx=$MPICXX"
  "$MPICXX" --version | awk 'NR <= 4'
} | tee "$OUT_DIR/provenance.txt"

set +e
"$HIPCC" "${FLAGS[@]}" "${MPI_COMPILE[@]}" "$SOURCE" -o \
  "$OUT_DIR/rank_transport" "${MPI_LINK[@]}" \
  >"$OUT_DIR/build.log" 2>&1
BUILD_RC=$?
set -e

if ((BUILD_RC != 0)); then
  printf '{"status":"compile_failed","exit_code":%d,"log":"%s"}\n' \
    "$BUILD_RC" "$OUT_DIR/build.log" | tee "$OUT_DIR/build-status.json"
  awk '/error:|fatal error:|undefined reference/ {print NR ":" $0; n++; if (n == 80) exit}' \
    "$OUT_DIR/build.log" >&2
  exit "$BUILD_RC"
fi

sha256sum "$OUT_DIR/rank_transport" | tee "$OUT_DIR/binary.sha256"
awk '
  /remark:.*(Function Name|SGPR|VGPR|AGPR|Scratch|LDS|Occupancy)/ ||
  /kernel resource usage/ {print}
' "$OUT_DIR/build.log" >"$OUT_DIR/resource-remarks.txt"

if ((WANT_ISA)); then
  set +e
  "$HIPCC" "${FLAGS[@]}" "${MPI_COMPILE[@]}" --cuda-device-only -S \
    "$SOURCE" -o "$OUT_DIR/rank_transport.s" \
    >"$OUT_DIR/isa.log" 2>&1
  ISA_RC=$?
  set -e
  if ((ISA_RC != 0)); then
    printf '{"status":"isa_failed","exit_code":%d,"log":"%s"}\n' \
      "$ISA_RC" "$OUT_DIR/isa.log" | tee "$OUT_DIR/build-status.json"
    exit "$ISA_RC"
  fi
  awk '
    /\.amdhsa_kernel/ {kernel=$2}
    /\.amdhsa_(next_free_vgpr|next_free_sgpr|group_segment_fixed_size|private_segment_fixed_size)/ {
      print kernel, $0
    }
  ' "$OUT_DIR/rank_transport.s" >"$OUT_DIR/isa-resources.txt"
fi

printf '{"status":"built","binary":"%s","resource_remarks":"%s","isa":%s}\n' \
  "$OUT_DIR/rank_transport" "$OUT_DIR/resource-remarks.txt" \
  "$([[ $WANT_ISA == 1 ]] && echo true || echo false)" \
  | tee "$OUT_DIR/build-status.json"
