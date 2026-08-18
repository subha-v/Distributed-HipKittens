#!/usr/bin/env bash
# ============================================================================
# M24 BUILD GATE (design section E / work-list G6) -- CPU-ONLY.
#
# Proves the DEFAULT build (every K0P6_M24_* macro at 0) is instruction-
# identical to the PRE-M24 body.  This is the discipline the exp_38 cliff
# bought: reachable-but-unused code near the M7 epilogue cost +726.9 us, so a
# default-arm .text perturbation must be impossible, not unlikely.
#
# It compiles TWO kernels through the SAME pin path with the SAME compile line
# and compares:
#   (a) the -Rpass-analysis=kernel-resource-usage tuple (SGPR/VGPR/AGPR/
#       ScratchSize/Occupancy/Spill/LDS), and
#   (b) the DISASSEMBLY of k0pf6gm_mps_mega, plus a sha256 of the raw .text.
# Both must be IDENTICAL.  Anything else is a gate failure and blocks every
# M24 arm.
#
# The two kernels:
#   pre  : k0pf6gm_device_tile_m15.hip at ${PRE_REF} (default a776ac75, the
#          commit before the M24 implementation), from `git show`.
#   post : the working tree's k0pf6gm_device_tile_m15.hip, all macros default.
#
# Usage (on the node):
#   bash m24_build_gate.sh [SRC_DIR] [OUT_DIR]
#     SRC_DIR : a Distributed-HipKittens checkout carrying the M24 edits
#               (default: ~/nightshift_m24_build/src)
#     OUT_DIR : scratch (default: ~/nightshift_m24_build)
#   env PRE_REF=<git ref>   the pre-M24 commit (default a776ac75)
#       K0_AMD_ROOT=...     amd-master checkout (default ~/amd-master)
#       K0PF_IMAGE=...      image used when hipcc is not on the host
#
# Writes ${OUT_DIR}/gate_output.txt.  Exit 0 = IDENTICAL, 1 = DIVERGED.
# GPUs are never touched: hipcc, llvm-objdump and llvm-objcopy are CPU work.
# ============================================================================
set -euo pipefail

SRC_DIR="${1:-$HOME/nightshift_m24_build/src}"
OUT_DIR="${2:-$HOME/nightshift_m24_build}"
AMD_ROOT="${K0_AMD_ROOT:-$HOME/amd-master}"
IMAGE="${K0PF_IMAGE:-rocm/atom-dev:vllm-latest}"
PRE_REF="${PRE_REF:-a776ac75}"

M15_REL="distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip"
M15="${SRC_DIR}/${M15_REL}"
GATE="${OUT_DIR}/gate"
PIN="${GATE}/pin"
LOG="${OUT_DIR}/gate_output.txt"

[[ -f "${M15}" ]] || { echo "missing ${M15}" >&2; exit 2; }
[[ -d "${AMD_ROOT}/HipKittens/include" ]] || {
  echo "missing ${AMD_ROOT}/HipKittens (set K0_AMD_ROOT)" >&2; exit 2; }

rm -rf "${GATE}"; mkdir -p "${PIN}/distributed-kernels/fused_moe" "${GATE}/out"
exec > >(tee "${LOG}") 2>&1

echo "=== M24 BUILD GATE ==="
echo "date        : $(date -u +%FT%TZ)"
echo "host        : $(hostname)"
echo "SRC_DIR     : ${SRC_DIR}"
echo "PRE_REF     : ${PRE_REF}"
echo "src HEAD    : $(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo '?')"
echo "src status  : $(git -C "${SRC_DIR}" status --porcelain "${M15_REL}" 2>/dev/null | tr '\n' ' ')"
echo "post sha256 : $(shasum -a 256 "${M15}" | awk '{print $1}')"

# The PRE body comes from git, so it cannot be a hand-made "clean copy" that
# quietly differs from the commit the review was written against.
git -C "${SRC_DIR}" show "${PRE_REF}:${M15_REL}" > "${GATE}/pre_m15.hip"
echo "pre sha256  : $(shasum -a 256 "${GATE}/pre_m15.hip" | awk '{print $1}')"
echo "pre bytes   : $(wc -c < "${GATE}/pre_m15.hip")"
echo "post bytes  : $(wc -c < "${M15}")"
if cmp -s "${GATE}/pre_m15.hip" "${M15}"; then
  echo "WARNING: pre and post sources are IDENTICAL -- the gate is vacuous."
fi
echo

# Everything except the m15 body is copied once and shared by both builds, so
# the ONLY difference between the two compiles is the body itself.
cp -a "${SRC_DIR}/distributed-kernels/fused_moe/." \
      "${PIN}/distributed-kernels/fused_moe/"
MPS="${PIN}/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"

mkpin() {   # $1 = body file.  ONE pin path for both builds: a path difference
            # is a codegen difference in ELF metadata and would muddy the
            # comparison.
  {
    echo "// M24 BUILD-GATE PIN (generated $(date -u +%FT%TZ))."
    echo "// Body below is k0pf6gm_device_tile_m15.hip VERBATIM (inline, not"
    echo "// #include: the mori JIT content hash does not hash -I paths)."
    echo "#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega"
    cat "$1"
  } > "${MPS}"
}

HIPCC_ARGS=(
  --offload-arch=gfx950
  -std=c++17 -O3 -c
  -DKITTENS_CDNA4
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math
  -Rpass-analysis=kernel-resource-usage
  -mllvm -amdgpu-mfma-vgpr-form=1
  -I"${AMD_ROOT}/HipKittens/include"
  -I"${AMD_ROOT}/HipKittens/prototype"
  -I"${AMD_ROOT}/auto-gpu-kernel/k0_fused_moe/solution/hip"
  -I"${AMD_ROOT}/auto-gpu-kernel/k0_fused_moe/prefill_opt/kernels"
  -I"${SRC_DIR}/include"
  -I"${PIN}/distributed-kernels/fused_moe"
)
for d in "${AMD_ROOT}/mori/include" "${SRC_DIR}/third_party/mori/include" \
         /opt/rocm/include; do
  [[ -d "${d}" ]] && HIPCC_ARGS+=(-I"${d}")
done

IN_DOCKER=0
command -v hipcc >/dev/null 2>&1 || IN_DOCKER=1

run_tool() {   # run a rocm-llvm tool on the host or in the image
  local tool="$1"; shift
  if [[ "${IN_DOCKER}" == "0" ]]; then
    local bin="${tool}"
    command -v "${bin}" >/dev/null 2>&1 || bin="/opt/rocm/llvm/bin/${tool}"
    "${bin}" "$@"
  else
    docker run --rm \
      -v "${SRC_DIR}:${SRC_DIR}:ro" -v "${AMD_ROOT}:${AMD_ROOT}:ro" \
      -v "${OUT_DIR}:${OUT_DIR}" \
      --entrypoint "/opt/rocm/llvm/bin/${tool}" "${IMAGE}" "$@"
  fi
}

build() {   # $1 = tag
  local tag="$1"
  local obj="${GATE}/out/${tag}.o"
  local res="${GATE}/out/${tag}.res"
  if [[ "${IN_DOCKER}" == "0" ]]; then
    hipcc "${HIPCC_ARGS[@]}" "${MPS}" -o "${obj}" 2> "${res}"
  else
    docker run --rm \
      -v "${SRC_DIR}:${SRC_DIR}:ro" -v "${AMD_ROOT}:${AMD_ROOT}:ro" \
      -v "${OUT_DIR}:${OUT_DIR}" \
      --entrypoint /opt/rocm/bin/hipcc "${IMAGE}" \
      "${HIPCC_ARGS[@]}" "${MPS}" -o "${obj}" 2> "${res}"
  fi
  # Resource tuple, normalised: drop the source path the remark prints.
  grep -E "SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS|Function Name" \
    "${res}" | sed -E 's/^[^ ]*:[0-9]+:[0-9]+: //' \
    > "${GATE}/out/${tag}.tuple" || true
  # Disassembly of the kernel, with addresses and raw encodings stripped so the
  # comparison is over INSTRUCTIONS, not over where they happen to land.
  run_tool llvm-objdump -d --mcpu=gfx950 "${obj}" \
    > "${GATE}/out/${tag}.dis.raw"
  sed -E 's/^[[:space:]]*[0-9a-f]+:([[:space:]]+[0-9a-f]{2})+[[:space:]]*//' \
    "${GATE}/out/${tag}.dis.raw" \
    | sed -E 's/\/\/ [0-9A-F]{12}: [0-9A-F]{8}.*$//' \
    | grep -v -E '^\s*$|file format|Disassembly of section' \
    > "${GATE}/out/${tag}.dis"
  run_tool llvm-objcopy -O binary --only-section=.text "${obj}" \
    "${GATE}/out/${tag}.text" 2>/dev/null || true
}

echo "--- building PRE (${PRE_REF}, pre-M24 body) ---"
mkpin "${GATE}/pre_m15.hip"
build pre
echo "--- building POST (working tree, all K0P6_M24_* at their 0 defaults) ---"
mkpin "${M15}"
build post

echo
echo "=== compile line (identical for both) ==="
printf '%s\n' "hipcc ${HIPCC_ARGS[*]} ${MPS}"

echo
echo "=== resource tuples ==="
echo "--- pre ---";  cat "${GATE}/out/pre.tuple"
echo "--- post ---"; cat "${GATE}/out/post.tuple"

rc=0
echo
echo "=== VERDICTS ==="
if diff -u "${GATE}/out/pre.tuple" "${GATE}/out/post.tuple" > "${GATE}/out/tuple.diff"; then
  echo "RESOURCE TUPLE : IDENTICAL"
else
  echo "RESOURCE TUPLE : DIVERGED"
  cat "${GATE}/out/tuple.diff"
  rc=1
fi

if [[ -s "${GATE}/out/pre.dis" && -s "${GATE}/out/post.dis" ]]; then
  if diff -u "${GATE}/out/pre.dis" "${GATE}/out/post.dis" > "${GATE}/out/dis.diff"; then
    echo "DISASSEMBLY    : IDENTICAL ($(wc -l < "${GATE}/out/post.dis") lines)"
  else
    echo "DISASSEMBLY    : DIVERGED"
    head -80 "${GATE}/out/dis.diff"
    echo "  (full diff: ${GATE}/out/dis.diff)"
    rc=1
  fi
else
  echo "DISASSEMBLY    : UNAVAILABLE -- llvm-objdump produced nothing."
  echo "                 The gate is NOT satisfied by the tuple alone."
  rc=1
fi

pre_text="$(shasum -a 256 "${GATE}/out/pre.text" 2>/dev/null | awk '{print $1}')"
post_text="$(shasum -a 256 "${GATE}/out/post.text" 2>/dev/null | awk '{print $1}')"
if [[ -n "${pre_text}" && "${pre_text}" == "${post_text}" ]]; then
  echo ".text sha256   : IDENTICAL ${pre_text}"
elif [[ -n "${pre_text}" && -n "${post_text}" ]]; then
  echo ".text sha256   : DIVERGED  pre=${pre_text} post=${post_text}"
  rc=1
else
  echo ".text sha256   : unavailable (llvm-objcopy); the disassembly compare above governs"
fi

echo
if [[ "${rc}" == "0" ]]; then
  echo "M24_BUILD_GATE_RECEIPT: PASS -- the default build is instruction-identical to ${PRE_REF}"
else
  echo "M24_BUILD_GATE_RECEIPT: FAIL -- the default build is NOT instruction-identical; every M24 arm is blocked"
fi
echo "evidence: ${GATE}/out/{pre,post}.{tuple,dis,text}, ${LOG}"
exit "${rc}"
