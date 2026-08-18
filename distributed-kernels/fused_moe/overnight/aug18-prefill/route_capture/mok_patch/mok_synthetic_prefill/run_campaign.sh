#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K0_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${K0_ROOT}/../.." && pwd)"
HIPKITTENS_ROOT="${HIPKITTENS_ROOT:-${WORKSPACE_ROOT}/HipKittens}"
DHK_ROOT="${DHK_ROOT:-${HOME}/Distributed-HipKittens}"

tag="${1:-mok_b4096_$(date -u +%Y%m%dT%H%M%SZ)}"
run_count="${2:-5}"
image="${K0PF_IMAGE:-rocm/atom-dev:vllm-latest}"
output_root="${K0_MOK_OUTPUT_ROOT:-${HOME}/k0-mok-synthetic-results/${tag}}"
cache_root="${K0_MOK_CACHE_ROOT:-${HOME}/.cache/k0-mok-synthetic-prefill}"
lock_dir="${K0_MOK_LOCK_DIR:-/tmp/k0_mok_synthetic_gpu_lock}"
arm_csv="${K0_MOK_ARMS:-production,pf4h}"
IFS=',' read -r -a base_arms <<< "${arm_csv}"

if [[ ! -f "${K0_ROOT}/CLAUDE.md" ]]; then
  echo "k0 root not found at ${K0_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${HIPKITTENS_ROOT}/include/kittens.cuh" ]]; then
  echo "HipKittens root not found at ${HIPKITTENS_ROOT}" >&2
  exit 3
fi
if [[ ",${arm_csv}," == *,mps_mega,* ]] &&
   [[ ! -f "${DHK_ROOT}/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" ]]; then
  echo "Distributed-HipKittens MPS source not found at ${DHK_ROOT}" >&2
  exit 3
fi
if ! [[ "${run_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "run_count must be a positive integer" >&2
  exit 4
fi
if [[ "${#base_arms[@]}" -lt 2 || ",${arm_csv}," != *,production,* ]]; then
  echo "K0_MOK_ARMS must contain production and at least one candidate" >&2
  exit 4
fi
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "required image is not present: ${image}" >&2
  exit 5
fi
if ! mkdir "${lock_dir}" 2>/dev/null; then
  echo "GPU lease is already held: ${lock_dir}" >&2
  exit 6
fi
cleanup() {
  rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "${output_root}" "${cache_root}/build" "${cache_root}/mori"
if [[ -e "${output_root}/summary.json" ]]; then
  echo "refusing to overwrite completed campaign ${output_root}" >&2
  exit 7
fi

assert_idle() {
  local process
  for process in torchrun mpirun orterun; do
    if pgrep -x "${process}" >/dev/null; then
      echo "refusing to run: ${process} is active" >&2
      pgrep -a -x "${process}" >&2 || true
      exit 20
    fi
  done
  local idle
  idle="$(
    /opt/rocm/bin/rocm-smi --showuse --csv 2>/dev/null |
      awk -F, '$1 ~ /^card[0-7]$/ && $2 == 0 {count++} END {print count + 0}'
  )"
  if [[ "${idle}" -ne 8 ]]; then
    echo "refusing to run: only ${idle}/8 GPUs report 0% use" >&2
    /opt/rocm/bin/rocm-smi --showuse --showmemuse --csv >&2 || true
    exit 21
  fi
}

for run in $(seq 1 "${run_count}"); do
  assert_idle
  run_dir="${output_root}/run${run}"
  log="${output_root}/run${run}.log"
  if [[ -e "${run_dir}" || -e "${log}" ]]; then
    echo "refusing to overwrite ${run_dir} or ${log}" >&2
    exit 22
  fi
  mkdir -p "${run_dir}"
  rotation=$(( (run - 1) % ${#base_arms[@]} ))
  ordered_arms=()
  for ((offset = 0; offset < ${#base_arms[@]}; ++offset)); do
    ordered_arms+=("${base_arms[$(( (rotation + offset) % ${#base_arms[@]} ))]}")
  done
  printf -v arms '%s,' "${ordered_arms[@]}"
  arms="${arms%,}"
  container_name="subvadla_k0_mok_${tag//[^a-zA-Z0-9_.-]/_}_r${run}"
  echo "starting run=${run}/${run_count} arms=${arms} at $(date -u +%FT%TZ)"

  set +e
  setsid -w timeout --signal=TERM "${K0_MOK_RUN_TIMEOUT:-2400}" \
    docker run --rm \
      --name "${container_name}" \
      --network=host --ipc=host \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      -v "${WORKSPACE_ROOT}:/workspace/amd-master:ro" \
      -v "${DHK_ROOT}:/workspace/Distributed-HipKittens:ro" \
      -v "${cache_root}:/work" \
      -v "${run_dir}:/out" \
      -e HSA_XNACK=1 \
      -e MORI_GPU_ARCHS=gfx950 \
      -e MORI_SHMEM_HEAP_SIZE=34359738368 \
      -e "K0_HOSTNAME=$(hostname)" \
      -e "K0_CONTAINER_IMAGE=${image}" \
      -e K0_BASE=/workspace/amd-master/auto-gpu-kernel/k0_fused_moe \
      -e K0_DHK_ROOT=/workspace/Distributed-HipKittens \
      -e K0_PYBIND_DIR=/work/build/python \
      -e K0_OUT_DIR=/out \
      -e K0_INPUT_MODE=mok_synthetic \
      -e K0_BENCHMARK_PROTOCOL=mok_eager \
      -e "K0_ARMS=${arms}" \
      -e "K0_T=${K0_T:-4096}" \
      -e K0_T_LOC_MAX=40960 \
      -e "K0_PADMAX=${K0_PADMAX:-263136}" \
      -e "K0_MAXTOK=${K0_T:-4096}" \
      -e "K0_MAXTOK_PROD=${K0_T:-4096}" \
      -e K0_BLK=128 \
      -e K0_WARP=16 \
      -e "K0_MOK_SEED_BASE=${K0_MOK_SEED_BASE:-1234}" \
      -e "K0_MOK_WARMUP_ITERS=${K0_MOK_WARMUP_ITERS:-500}" \
      -e "K0_MOK_TIMED_ITERS=${K0_MOK_TIMED_ITERS:-100}" \
      -e "K0_MOK_ABSOLUTE_TOLERANCE=${K0_MOK_ABSOLUTE_TOLERANCE:-0.1}" \
      -e "K0_MOK_RELATIVE_TOLERANCE=${K0_MOK_RELATIVE_TOLERANCE:-0.1}" \
      -e "K0_MOK_POISON_OUT=${K0_MOK_POISON_OUT:-1}" \
      -e "K0_MOK_POISON_SELFTEST=${K0_MOK_POISON_SELFTEST:-1}" \
      -e "K0_MOK_WEIGHT_CHUNK_ELEMENTS=${K0_MOK_WEIGHT_CHUNK_ELEMENTS:-33554432}" \
      -e "K0_PF5_PRIMITIVE_ONLY=${K0_PF5_PRIMITIVE_ONLY:-0}" \
      -e "K0_PF6GM_G=${K0_PF6GM_G:-3}" \
      -e "K0_MPS_CFG=${K0_MPS_CFG:-}" \
      -e "K0_MOK_ROUTE_HIST=${K0_MOK_ROUTE_HIST:-}" \
      -e "K0_MOK_REP_EXPERTS=${K0_MOK_REP_EXPERTS:-}" \
      -e "K0_MOK_REP_THRESHOLD=${K0_MOK_REP_THRESHOLD:-0}" \
      -e "K0_MOK_M20=${K0_MOK_M20:-}" \
      -e "K0_M15B=${K0_M15B:-}" \
      -e "K0_SYNTH_ROUTE=${K0_SYNTH_ROUTE:-}" \
      -e "K0_SYNTH_SEED=${K0_SYNTH_SEED:-0}" \
      -e "K0_MPS_DEBUG_STOP=${K0_MPS_DEBUG_STOP:-}" \
      -e "K0_MPS_DESC_DUMP=${K0_MPS_DESC_DUMP:-}" \
      -e "K0_MPS_TRACE=${K0_MPS_TRACE:-}" \
      -e "K0_MPS_SOAK_ITERS=${K0_MPS_SOAK_ITERS:-600}" \
      -e "K0_SPIN_LIMIT=${K0_SPIN_LIMIT:-2000000}" \
      -e "K0_PF6_DEBUG_PHASE=${K0_PF6_DEBUG_PHASE:-0}" \
      -v "${cache_root}/mori:/root/.mori" \
      --entrypoint bash \
      "${image}" -lc \
      'set -euo pipefail
       PYTHON=/opt/venv/bin/python
       [[ -x "${PYTHON}" ]] || PYTHON=python3
       cmake -S "${K0_BASE}/benchmarks/mok_synthetic_prefill" -B /work/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
         -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/work/build/python \
         -DK0_ROOT="${K0_BASE}" \
         -DHIPKITTENS_ROOT=/workspace/amd-master/HipKittens \
         -DDHK_ROOT="${K0_DHK_ROOT}" \
         -DPython3_EXECUTABLE="${PYTHON}"
       cmake --build /work/build -j"${K0_MOK_BUILD_JOBS:-16}"
       cd "${K0_BASE}"
       "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
         prefill_opt/host/e004pf_k0pf_ab.py' \
      2>&1 | tee "${log}"
  run_rc="${PIPESTATUS[0]}"
  set -e

  shopt -s nullglob
  rank_files=("${run_dir}"/k0pf_mok_synthetic_rank*.json)
  shopt -u nullglob
  if [[ "${#rank_files[@]}" -ne 8 ]]; then
    echo "run ${run} did not produce eight rank JSON files" >&2
    exit 23
  fi
  if [[ "${run_rc}" -ne 0 ]]; then
    python3 "${SCRIPT_DIR}/summarize.py" "${run_dir}" \
      --output "${output_root}/summary.json"
    echo "campaign blocked before timing: ${output_root}/summary.json" >&2
    exit "${run_rc}"
  fi
  echo "completed run=${run}/${run_count} at $(date -u +%FT%TZ)"
done

run_dirs=()
for run in $(seq 1 "${run_count}"); do
  run_dirs+=("${output_root}/run${run}")
done
python3 "${SCRIPT_DIR}/summarize.py" "${run_dirs[@]}" \
  --output "${output_root}/summary.json"
echo "campaign complete: ${output_root}/summary.json"
