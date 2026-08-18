#!/usr/bin/env bash
# ============================================================================
# M24 compile check — CPU-ONLY, safe while the GPUs are running a measurement.
#
# Compiles the M24 kernel body in BOTH configurations that matter:
#   base : every K0P6_M24_* macro at its default 0   (must be the pre-M24 body)
#   fill : K0P6_M24_FILL=1 K0P6_M24_ZERO_PAD=1       (the arm shape)
# and, optionally, the composed / MoK-source variants.
#
# It builds each configuration as a RUN PIN, exactly as DECOMP_RUNBOOK.md 4.2
# prescribes: the pin's k0pf6gm_device_tile_mps.hip is the kernel-name stanza
# plus k0pf6gm_device_tile_m15.hip INLINED VERBATIM (never #include -- the mori
# JIT content hash ignores -I paths, the exp_04 trap).
#
# This is NOT the build gate. It produces .o + resource-usage reports so an
# implementation error surfaces before any GPU time; the byte-identical .text
# comparison (design section E / work-list G6) is a separate, later step.
#
# Usage (on the node):
#   bash m24_compile_check.sh [SRC_DIR] [OUT_DIR]
#     SRC_DIR: a Distributed-HipKittens checkout carrying the M24 edits
#              (default: ~/nightshift_m24_build/src)
#     OUT_DIR: scratch                (default: ~/nightshift_m24_build)
#
# Compiles on the host if hipcc is present, otherwise inside the same image the
# MoK campaign uses (rocm/atom-dev:vllm-latest, K0PF_IMAGE to override) with no
# GPU devices attached -- hipcc is CPU-only work.
# ============================================================================
set -euo pipefail

SRC_DIR="${1:-$HOME/nightshift_m24_build/src}"
OUT_DIR="${2:-$HOME/nightshift_m24_build}"
AMD_ROOT="${K0_AMD_ROOT:-$HOME/amd-master}"
IMAGE="${K0PF_IMAGE:-rocm/atom-dev:vllm-latest}"

M15="${SRC_DIR}/distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip"
[[ -f "${M15}" ]] || { echo "missing ${M15}" >&2; exit 2; }
[[ -d "${AMD_ROOT}/HipKittens/include" ]] || {
  echo "missing ${AMD_ROOT}/HipKittens (set K0_AMD_ROOT)" >&2; exit 2; }

mkdir -p "${OUT_DIR}/pins" "${OUT_DIR}/obj"

mkpin() {   # $1 = pin name, $2.. = extra "MACRO value" defines
  local name="$1"; shift
  local dst="${OUT_DIR}/pins/${name}"
  rm -rf "${dst}"; mkdir -p "${dst}/distributed-kernels/fused_moe"
  cp -a "${SRC_DIR}/distributed-kernels/fused_moe/." \
        "${dst}/distributed-kernels/fused_moe/"
  local mps="${dst}/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
  {
    echo "// M24 COMPILE-CHECK PIN (generated $(date -u +%FT%TZ))."
    echo "// Body below is k0pf6gm_device_tile_m15.hip VERBATIM (inline, not"
    echo "// #include: the mori JIT content hash does not hash -I paths)."
    echo "#define K0P6_M15_KERNEL_NAME k0pf6gm_mps_mega"
    local d
    for d in "$@"; do echo "#define ${d}"; done
    cat "${M15}"
  } > "${mps}"
  echo "${mps}"
}

# The MoK modules' own compile options (benchmarks/mok_synthetic_prefill/
# CMakeLists.txt configure_k0_module) plus the include set the mps tile needs.
hip_compile() {   # $1 = pin source, $2 = tag
  local src="$1" tag="$2"
  local args=(
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
    -I"${SRC_DIR}/distributed-kernels/fused_moe"
  )
  # mori headers, if the checkout carries them
  for d in "${AMD_ROOT}/mori/include" "${SRC_DIR}/third_party/mori/include" \
           /opt/rocm/include; do
    [[ -d "${d}" ]] && args+=(-I"${d}")
  done
  args+=("${src}" -o "${OUT_DIR}/obj/${tag}.o")
  if command -v hipcc >/dev/null 2>&1; then
    echo "+ hipcc ${args[*]}" | tee "${OUT_DIR}/obj/${tag}.cmd"
    hipcc "${args[@]}" 2> "${OUT_DIR}/obj/${tag}.res" || return 1
  else
    echo "+ [docker ${IMAGE}] /opt/rocm/bin/hipcc ${args[*]}" \
      | tee "${OUT_DIR}/obj/${tag}.cmd"
    docker run --rm \
      -v "${SRC_DIR}:${SRC_DIR}:ro" -v "${AMD_ROOT}:${AMD_ROOT}:ro" \
      -v "${OUT_DIR}:${OUT_DIR}" \
      --entrypoint /opt/rocm/bin/hipcc "${IMAGE}" \
      "${args[@]}" 2> "${OUT_DIR}/obj/${tag}.res" || return 1
  fi
}

run_cfg() {   # $1 = tag, $2.. = defines
  local tag="$1"; shift
  local pin; pin="$(mkpin "${tag}" "$@")"
  echo "=== ${tag}: $*"
  if hip_compile "${pin}" "${tag}"; then
    echo "--- ${tag} OK"
    grep -E "SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spill|LDS" \
      "${OUT_DIR}/obj/${tag}.res" || true
  else
    echo "--- ${tag} FAILED"
    tail -40 "${OUT_DIR}/obj/${tag}.res"
    return 1
  fi
}

rc=0
run_cfg base                                                        || rc=1
run_cfg fill  "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1"               || rc=1
# M24_FULL DEFAULTS TO 1 (rev 3 review fix R.7d). It used to default to 0, so
# the default run built only `base` and `fill` and reported success -- while the
# two constructs most likely to fail codegen (the inline-asm opacity barrier on
# a non-uniform value, and the NORIG_TABLE local array under divergent control
# flow) live ONLY in the gated pins, as does R0b's blocking inert_4096 pin and
# arm D's mandatory skew pin. The first time anyone would have learned that the
# blocking pin does not build is with the node claimed and the GPU window open.
# This is hipcc, i.e. CPU-only work; there is no reason to skip it. Set
# M24_FULL=0 only to shorten a smoke check.
if [[ "${M24_FULL:-1}" != "0" ]]; then
  run_cfg fill_nullwork "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_NULLWORK 1"                       || rc=1
  run_cfg fill_strict   "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_STRICT 1"                         || rc=1
  run_cfg inert_4096    "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_NORIG_CONST 4096"                 || rc=1
  run_cfg fill_1539     "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_NORIG_CONST 1539"                 || rc=1
  run_cfg skew_a        "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_NORIG_TABLE {4096,0,0,0,0,0,0,0}" || rc=1
  run_cfg ladder_0      "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M24_NORIG_TABLE {0,0,0,0,0,0,0,0}"    || rc=1
  run_cfg dirty         "K0P6_M24_FILL 1" "K0P6_M24_ALLOW_DIRTY_PAD 1" || rc=1
  # Compositions section H.8 claims. The descriptor cascade + the raise-not-
  # redefine block must survive each of them with no macro-redefinition
  # warning and no slot collision.
  run_cfg fill_staged   "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M15_STAGED 1"                         || rc=1
  run_cfg fill_rep      "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M15_REPLICATE 1"                      || rc=1
  run_cfg fill_adapt    "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M15_REPLICATE 1" "K0P6_M15_ADAPTIVE 1" || rc=1
  run_cfg fill_slotpool "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                        "K0P6_M15_REPLICATE 1" "K0P6_M15_ADAPTIVE 1" \
                        "K0P6_M20_SLOTPOOL 1"                       || rc=1
fi
# The NULLWORK-alone and STRICT-alone builds MUST fail (their #error
# cross-checks). A silent success here means a guard was lost.
neg_cfg() {   # $1 = tag, $2.. = defines that must NOT compile
  local tag="$1"; shift
  local pin; pin="$(mkpin "neg_${tag}" "$@")"
  echo "=== neg_${tag} (must FAIL): $*"
  if hip_compile "${pin}" "neg_${tag}"; then
    echo "--- neg_${tag} COMPILED, but the #error cross-check should have fired"
    return 1
  fi
  if grep -q "K0P6_M24" "${OUT_DIR}/obj/neg_${tag}.res"; then
    echo "--- neg_${tag} correctly refused"
    return 0
  fi
  echo "--- neg_${tag} failed for the WRONG reason:"
  tail -20 "${OUT_DIR}/obj/neg_${tag}.res"
  return 1
}
if [[ "${M24_FULL:-1}" != "0" ]]; then
  neg_cfg nullwork_alone "K0P6_M24_NULLWORK 1"                      || rc=1
  neg_cfg strict_alone   "K0P6_M24_STRICT 1"                        || rc=1
  neg_cfg fill_nozero    "K0P6_M24_FILL 1"                          || rc=1
  neg_cfg both_sources   "K0P6_M24_FILL 1" "K0P6_M24_ZERO_PAD 1" \
                         "K0P6_M24_NORIG_CONST 1024" \
                         "K0P6_M24_NORIG_TABLE {0,0,0,0,0,0,0,0}"   || rc=1
fi
echo "resource reports in ${OUT_DIR}/obj/*.res"
exit "${rc}"
