#!/usr/bin/env bash
set -euo pipefail

# M15 serving campaign driver (v3).  Contract: CAMPAIGN.md in this directory.
#
# v3 = v2 plus the round-robin expert-placement arms (<base>_rr).  Every
# non-_rr arm is byte-compatible with v2: RR_PATCH is empty by default, so no
# mount, no chain step, no env and no serve flag are added.
#
# NOT YET EXECUTED.  Derived from the only executable PF4H recipe,
# benchmarks/2026-07-31_pf4h_variance/run_order_balanced_ab.sh, extended
# additively with:
#
#   * the m15 arm (VLLM_PF4H_INTEGRATION_MODE=m15, one fused launch)
#   * the tuned RCCL baseline arm (never a strawman; see --arm rccl)
#   * --profile-pass   : an 8-rank torch-profiler pass, its own server
#                        lifetime, never a latency result
#   * --skew-pass      : an untimed expert-histogram / per-rank-count pass
#   * --bundle         : the self-contained replay bundle
#   * interactive cells c8/c16/c32 alongside the batch c512/c1024 cells
#   * the stock-only 16384 batching-triangle caveat cell
#
# Arm order alternates AB/BA across pairs and workload order alternates, so
# neither arm nor workload can inherit a warm cache position.

readonly RUN_TAG="${RUN_TAG:-m15r1}"
readonly DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
readonly ROOT="${ROOT:-/home/subvadla/${DATE_TAG}_m15_campaign_${RUN_TAG}}"
readonly PACKET="${PACKET:?set PACKET to the deployed amd-master packet checkout}"
readonly N2="${N2:?set N2 to the exact k0_n2*.so}"
readonly M15_SOURCES="${M15_SOURCES:?set M15_SOURCES to the staged deploy dir}"
readonly DHK_ROOT="${DHK_ROOT:?set DHK_ROOT to the Distributed-HipKittens checkout}"
readonly CLIENT="${CLIENT:?set CLIENT to bench_exact_token_ids_v2.py}"
readonly MODEL="${MODEL:-/hf_home/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52}"
readonly IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.25.1}"
readonly READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
readonly STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-180}"
readonly MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
# Real-text prompts. "qsl" replays the official MLPerf DeepSeek-R1 set so
# expert routing carries production-like popularity skew; "synthetic" is the
# original uniform-random token stream and remains the default.
readonly PROMPT_SOURCE="${PROMPT_SOURCE:-synthetic}"
readonly QSL_PKL="${QSL_PKL:-}"
# If DHK_ROOT is a git WORKTREE its .git is a file pointing at the parent
# repo's gitdir, so apply.py's HEAD check needs that path present at the
# same absolute location inside the container.  Without it git reports
# "not a git repository", HEAD reads None and the install refuses.
DHK_GITDIR=""
if [[ -f "$DHK_ROOT/.git" ]]; then
    DHK_GITDIR="$(sed -n 's/^gitdir: //p' "$DHK_ROOT/.git" | head -1)"
    DHK_GITDIR="${DHK_GITDIR%%/worktrees/*}"
fi
readonly DHK_GITDIR
readonly GPU_CLAIM_FILE="${GPU_CLAIM_FILE:-$HOME/GPU_CLAIM}"
readonly GPU_CLAIM_NAME="${GPU_CLAIM_NAME:-m15pkt}"

# --- EPLB extension (2026-08-17) ---------------------------------------------
# Any arm named <base>_eplb serves with vLLM EPLB enabled on top of the
# unchanged <base> environment.  num_redundant_experts=128 = 16 extra
# expert slots per GPU = the same nominal budget as M18/M19 top-16
# replication, so stock_eplb is the production-native balanced baseline.
EPLB_JSON_DEFAULT='{"window_size": 400, "step_interval": 400, "num_redundant_experts": 128, "log_balancedness": true, "log_balancedness_interval": 50, "use_async": true}'
readonly EPLB_JSON="${EPLB_JSON:-$EPLB_JSON_DEFAULT}"
# Coverage instrumentation (applied to EVERY arm, after apply.py): counts
# per-step exact-B4096 seals so each arm reports token-weighted megakernel
# coverage.  See coverage_patch.py.
readonly COV_PATCH="${COV_PATCH:-$HOME/eplb_campaign/coverage_patch.py}"
# Optional M23 ragged-seal patch (post-apply, like COV_PATCH).  Runtime-inert
# on stock-target servers; on pf4h-target servers it relaxes the exact-4096
# unanimity seal to the padded (512,4096] bucket production already serves.
readonly M23_PATCH="${M23_PATCH:-}"
# Optional round-robin expert-placement patch (post-apply, after the M23
# chain).  Runtime-inert unless VLLM_PF4H_RR_PLACEMENT=1, which only the
# <base>_rr arms set.  See rr_patch.py / RR_PLACEMENT_NOTES.md: round-robin
# is realised as a static permutation (weight loader + one router gather), so
# physical ownership stays contiguous rank-major and every downstream
# consumer -- MoRI dispatch, AITER, the megakernel -- is the linear path.
readonly RR_PATCH="${RR_PATCH:-}"
# Identical unmeasured prewarm pass for BOTH arms (seed offset +777) so the
# EPLB arm has load history / rearrangements before the measured cell without
# giving either arm an asymmetric warm-cache position.
readonly EPLB_PREWARM="${EPLB_PREWARM:-1}"
# Host path of the pinned campaign dir, for the summarize/skew helpers that
# the original script located via $(dirname $0).
readonly CAMPAIGN_SRC_DIR="${CAMPAIGN_SRC_DIR:-$PACKET/benchmarks/2026-08-12_m15_campaign}"

# Arm-name suffixes are stripped in order: <base>[_eplb][_rr].  base_arm must
# strip BOTH so arm_env, apply_command and required_receipts keep resolving
# the underlying arm (m15_rr -> m15).
base_arm() { local a="${1%_rr}"; printf '%s' "${a%_eplb}"; }
arm_is_eplb() { [[ "$1" == *_eplb ]]; }
arm_is_rr() { [[ "$1" == *_rr ]]; }
# -----------------------------------------------------------------------------

PROFILE_PASS=0
SKEW_PASS=0
MAKE_BUNDLE=0
PAIRS="${PAIRS:-4}"
ARMS="${ARMS:-stock,m15}"
CELLS="${CELLS:-c16,c32,c512,c1024}"
CURRENT_CONTAINER=""

usage() {
    cat <<'USAGE'
run_m15_campaign.sh [options]
  --arms a,b        arms to run: stock, pf4h, m15, m18, m19, m20, rccl   (default stock,m15)
                    suffix _eplb = the same arm with vLLM EPLB enabled
                    suffix _rr   = the same arm with round-robin expert placement
                                   (VLLM_PF4H_RR_PLACEMENT=1 +
                                    --expert-placement-strategy round_robin;
                                    needs RR_PATCH=/path/to/rr_patch.py)
                    m18 = m15 + static hot-expert replication (M18_REP_EXPERTS env overrides the measured top-16 set)
                    m20 = m19 routing + budgeted replica cache (M20_BUDGET_LAYERS = cached model-layer csv, required)
  --cells c,...     c8 c16 c32 c512 c1024 stock16384           (default c16,c32,c512,c1024)
  --pairs N         order-balanced pairs                       (default 4)
  --profile-pass    add the 8-rank profiler pass (own lifetime, untimed)
  --skew-pass       add the untimed expert-skew diagnostic pass
  --bundle          write the replay bundle after the last pair
USAGE
}

while (($#)); do
    case "$1" in
        --arms) ARMS="$2"; shift 2 ;;
        --cells) CELLS="$2"; shift 2 ;;
        --pairs) PAIRS="$2"; shift 2 ;;
        --profile-pass) PROFILE_PASS=1; shift ;;
        --skew-pass) SKEW_PASS=1; shift ;;
        --bundle) MAKE_BUNDLE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- GPU claim

claim_gpus() {
    if [[ -e "$GPU_CLAIM_FILE" ]] \
        && [[ "$(cat "$GPU_CLAIM_FILE")" != "$GPU_CLAIM_NAME" ]]; then
        echo "BLOCKED: GPUs claimed by '$(cat "$GPU_CLAIM_FILE")'" >&2
        return 1
    fi
    # Count only real process rows: rocm-smi --showpids emits banner text
    # ("KFD process information:") that no keyword filter reliably strips, so
    # anchor on a leading PID instead.  The previous grep-based form matched
    # that banner line and refused unconditionally.
    local foreign
    foreign=$(rocm-smi --showpids 2>/dev/null \
        | awk '/^[0-9]+/ && $2 != "gpuagent" {c++} END{print c+0}')
    if [[ "$foreign" -ne 0 ]]; then
        echo "BLOCKED: $foreign foreign GPU process(es) present" >&2
        rocm-smi --showpids >&2
        return 1
    fi
    printf '%s' "$GPU_CLAIM_NAME" >"$GPU_CLAIM_FILE"
}

release_gpus() {
    if [[ -e "$GPU_CLAIM_FILE" ]] \
        && [[ "$(cat "$GPU_CLAIM_FILE")" == "$GPU_CLAIM_NAME" ]]; then
        rm -f "$GPU_CLAIM_FILE"
    fi
}

# ------------------------------------------------------------- server plumbing

is_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

count_log() { docker logs "$1" 2>&1 | grep -cF "$2" || true; }

stop_server() {
    local container="$1" deadline
    is_running "$container" || return 0
    # Never SIGKILL a GPU process.
    docker exec "$container" kill -INT 1 >/dev/null 2>&1 || true
    deadline=$((SECONDS + STOP_TIMEOUT_S))
    while is_running "$container" && ((SECONDS < deadline)); do sleep 5; done
    if is_running "$container"; then
        echo "BLOCKED: $container did not stop; no SIGKILL issued" >&2
        return 1
    fi
}

# Stop whatever server a failed pass left running, so the next pass can claim
# the GPUs and the name does not collide.
salvage_container() {
    if [[ -n "$CURRENT_CONTAINER" ]]; then
        if is_running "$CURRENT_CONTAINER"; then
            stop_server "$CURRENT_CONTAINER" || true
        fi
        CURRENT_CONTAINER=""
    fi
}

cleanup_on_exit() {
    local rc=$?
    [[ -n "$CURRENT_CONTAINER" ]] && is_running "$CURRENT_CONTAINER" \
        && stop_server "$CURRENT_CONTAINER" || true
    release_gpus
    exit "$rc"
}
trap cleanup_on_exit EXIT

# Environment blocks. One variable per experiment: the MORI arms keep the
# frozen PF4H fairness environment untouched, and the RCCL arm gets the
# report-validated tuning so the baseline is honest.
arm_env() {
    # Each element must be ONE argv token: docker's shorthand parser reads
    # "-e FOO=bar" as a single token and takes the value as " FOO=bar",
    # creating a variable whose name has a leading space -- so the real one
    # is unset and the serve line expanded --all2all-backend to ''.
    local arm
    arm="$(base_arm "$1")"
    case "$arm" in
        native)
            # TRUE PRODUCTION: untouched image, no PF4H env, no patches, no
            # apply.py — the deployment users actually run (MoRI + AITER,
            # generic <=512 graph captures, heavy prefill steps eager).
            printf '%s\n' \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput"
            ;;
        stock)
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=m15" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=stock" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput"
            ;;
        pf4h)
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=full" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=pf4h" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput"
            ;;
        m15)
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=m15" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=pf4h" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput"
            ;;
        m18)
            # The m15 environment plus the replication opt-in.  Default set =
            # the measured 20260814 aggregate top-16 (58.7% of routed traffic;
            # kernel-level 0.2489x production under the replayed histogram).
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=m15" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=pf4h" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput" \
                "-e" "VLLM_PF4H_M18_REP_EXPERTS=${M18_REP_EXPERTS:-6,5,4,2,0,3,1,7,8,9,20,10,11,19,17,16}"
            ;;
        m19)
            # m18 plus the per-chunk adaptive decision (theta = own routed-slot
            # count needed to keep a replicated expert local this chunk).
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=m15" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=pf4h" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput" \
                "-e" "VLLM_PF4H_M18_REP_EXPERTS=${M18_REP_EXPERTS:-/m15src/m18_layer_sets.json}" \
                "-e" "VLLM_PF4H_M19_THRESHOLD=${M19_THRESHOLD:-64}"
            ;;
        m20)
            # m19's routing machinery with the replica CACHE restricted to a
            # host-budgeted layer subset (M20_BUDGET_LAYERS = csv of model
            # layer ids) and the per-chunk decision computed BEFORE launch
            # (deletes m19's in-kernel pre-pass).  Uncached layers run the
            # plain m15 kernel on stock weights.
            printf '%s\n' \
                "-e" "VLLM_PF4H_INTEGRATION_MODE=m15" \
                "-e" "VLLM_PF4H_B4096_GRAPH_TARGET=pf4h" \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput" \
                "-e" "VLLM_PF4H_M18_REP_EXPERTS=${M18_REP_EXPERTS:-/m15src/m18_layer_sets.json}" \
                "-e" "VLLM_PF4H_M19_THRESHOLD=${M19_THRESHOLD:-64}" \
                "-e" "VLLM_PF4H_M20_BUDGET_LAYERS=${M20_BUDGET_LAYERS:?set M20_BUDGET_LAYERS to the cached model-layer csv}"
            ;;
        rccl)
            # Tuned RCCL baseline (CAMPAIGN.md section 7).  These two
            # variables belong ONLY to this arm.
            printf '%s\n' \
                "-e" "VLLM_ALL2ALL_BACKEND=allgather_reducescatter" \
                "-e" "NCCL_MIN_NCHANNELS=112" \
                "-e" "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
            ;;
        *) echo "unknown arm $arm" >&2; return 1 ;;
    esac
}

# The M15 arms need the megakernel installed; stock/pf4h reuse the four-kernel
# install.  --integration-mode m15 is a superset: it also installs the PF4H
# packet, so a stock-target server built this way is byte-identical in
# behaviour to one built with --integration-mode full.
apply_command() {
    local arm
    arm="$(base_arm "$1")"
    # apply.py reads --dhk-root's git HEAD.  The container is root over a
    # user-owned bind mount, so without safe.directory git refuses with
    # "dubious ownership", HEAD reads back as None, and the install aborts
    # with "Distributed-HipKittens HEAD is None".  The subshell keeps the
    # caller's && chain keyed on apply.py's own exit status.
    local base="( git config --global --add safe.directory '*' >/dev/null 2>&1; \
        python /packet/shim/pf4h_integration/apply.py \
        --site-packages /usr/local/lib/python3.12/dist-packages \
        --deploy-sources /packet/benchmarks/2026-07-29_pf4h_stage0/deploy_sources \
        --n2-extension /n2/k0_n2.cpython-312-x86_64-linux-gnu.so"
    if [[ "$arm" == "native" ]]; then
        printf 'true'   # true-production arm: untouched image, no install
        return 0
    fi
    if [[ "$arm" == "m15" || "$arm" == "m18" || "$arm" == "m19" || "$arm" == "m20" || "$arm" == "stock" ]]; then
        printf '%s --integration-mode m15 --m15-deploy-sources /m15src --dhk-root /dhk --apply )' "$base"
    elif [[ "$arm" == "pf4h" ]]; then
        printf '%s --apply )' "$base"
    else
        printf 'true'   # the rccl arm runs the untouched image
    fi
}

launch_server() {
    local container="$1" arm="$2" output_dir="$3" profile="${4:-0}"
    local -a env_args=()
    mapfile -t env_args < <(arm_env "$arm")
    # The RR arms carry the placement gate on TOP of their base environment,
    # so stock_rr differs from stock by exactly this variable plus the serve
    # flag below.
    if arm_is_rr "$arm"; then
        env_args+=("-e" "VLLM_PF4H_RR_PLACEMENT=1")
    fi
    local -a profile_args=()
    if ((profile)); then
        profile_args=(-e "VLLM_TORCH_PROFILER_DIR=/prof/traces")
    fi
    # native arm runs the SHIPPED defaults verbatim (per operator decision):
    # untouched image, default capture ladder (graphs to 256), heavy prefill
    # steps compiled-but-ungraphed exactly as the vendor delivers it.
    local native_args=""
    local eplb_args=""
    if arm_is_eplb "$arm"; then
        eplb_args=" --enable-eplb --eplb-config '$EPLB_JSON'"
    fi
    # --expert-placement-strategy round_robin is what makes the loader's disk
    # filter fetch {r, r+8, ...} per rank; rr_patch.py makes the weight loader
    # and the router agree with that set.  The two MUST ship together --
    # rr_patch refuses the flag without VLLM_PF4H_RR_PLACEMENT=1 because stock
    # vLLM silently loads the wrong experts in that combination.
    local rr_args=""
    if arm_is_rr "$arm"; then
        if [[ -z "$RR_PATCH" ]]; then
            echo "ERROR: arm $arm needs RR_PATCH set to rr_patch.py" >&2
            return 1
        fi
        rr_args=" --expert-placement-strategy round_robin"
    fi
    local patch_chain="python /covpatch/coverage_patch.py && "
    if [[ "$(base_arm "$arm")" == "native" ]]; then
        # true-production arm: pristine image, zero patch-chain steps
        patch_chain=""
    fi
    local m23_chain=""
    if [[ -n "$M23_PATCH" && -z "$patch_chain" ]]; then
        : # native arm: no patch chain at all
    elif [[ -n "$M23_PATCH" ]]; then
        m23_chain="python /covpatch/m23_patch.py && "
    fi
    local rr_chain=""
    if [[ -n "$RR_PATCH" && -n "$patch_chain" ]]; then
        rr_chain="python /covpatch/rr_patch.py && "
    fi

    # shellcheck disable=SC2046
    docker run -d \
        --name "$container" \
        --network host --ipc host --shm-size 32g \
        --device /dev/kfd --device /dev/dri \
        --group-add video --group-add render \
        "${env_args[@]}" "${profile_args[@]}" \
        -e VLLM_PF4H_ACTIVATION_FILE=/tmp/vllm-pf4h-enable \
        -e VLLM_ROCM_USE_AITER=1 \
        -e VLLM_ROCM_USE_AITER_MOE=1 \
        -e MORI_SHMEM_MODE=ISOLATION \
        -e MORI_GPU_ARCHS=gfx950 \
        -e HIP_FORCE_DEV_KERNARG=1 \
        -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e PYTORCH_NVML_BASED_CUDA_CHECK=1 \
        -v /data/hf_home:/hf_home:ro \
        -v "$PACKET:/packet:ro" \
        -v "$M15_SOURCES:/m15src:ro" \
        -v "$DHK_ROOT:/dhk:ro" \
        -v "$N2:/n2/k0_n2.cpython-312-x86_64-linux-gnu.so:ro" \
        ${QSL_PKL:+-v "$QSL_PKL:/qsl/dataset.pkl:ro"} \
        ${DHK_GITDIR:+-v "$DHK_GITDIR:$DHK_GITDIR:ro"} \
        ${M15_SKEW_HOOK:+-v "$M15_SKEW_HOOK:/skewhook:ro"} \
        ${M15_SKEW_HOOK:+-e PYTHONPATH=/skewhook} \
        ${M15_SKEW_HOOK:+-e M15_SKEW_OUT=/results/skew} \
        ${M15_SKEW_HOOK:+-e "M15_SKEW_CAPTURE_LAYERS=${M15_SKEW_CAPTURE_LAYERS:-}"} \
        ${M15_SKEW_HOOK:+-e "M15_SKEW_CAPTURE_MAX=${M15_SKEW_CAPTURE_MAX:-64}"} \
        -v "$COV_PATCH:/covpatch/coverage_patch.py:ro" \
        ${M23_PATCH:+-v "$M23_PATCH:/covpatch/m23_patch.py:ro"} \
        ${RR_PATCH:+-v "$RR_PATCH:/covpatch/rr_patch.py:ro"} \
        -v "$output_dir:/results" \
        -v "$output_dir:/prof" \
        --entrypoint bash \
        "$IMAGE" \
        -lc "test ! -e /tmp/vllm-pf4h-enable && \
             $(apply_command "$arm") && \
             ${patch_chain}${m23_chain}${rr_chain}exec vllm serve '$MODEL' \
               --served-model-name r1 \
               --host 0.0.0.0 --port 8000 \
               --tensor-parallel-size 1 \
               --data-parallel-size 8 \
               --api-server-count 8 \
               --enable-expert-parallel \
               --all2all-backend \"\${VLLM_ALL2ALL_BACKEND}\" \
               --kv-cache-dtype fp8 \
               --gpu-memory-utilization 0.70 \
               --max-model-len 32768 \
               --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
               --max-num-seqs 128 \
               --scheduling-policy fcfs \
               --enable-prefix-caching \
               --enable-chunked-prefill \
               --optimization-level 2 \
               --performance-mode balanced \
               --seed 0${eplb_args}${rr_args}${native_args}" >/dev/null
}

# Receipt gates (CAMPAIGN.md section 8). A missing receipt discards the cell.
required_receipts() {
    case "$(base_arm "$1")" in
        m15)
            printf '%s\n' \
                M15_SELECTION_RECEIPT M15_ACTIVATION_RECEIPT \
                M15_SYMMETRIC_OFFSET_RECEIPT M15_HEAP_DESCRIPTOR \
                M15_GSTACK_RECEIPT M15_GRAPH_COMMIT_RECEIPT \
                PF4H_GRAPH_OPERATOR_GATE
            ;;
        m18)
            printf '%s\n' \
                M15_SELECTION_RECEIPT M15_ACTIVATION_RECEIPT \
                M15_SYMMETRIC_OFFSET_RECEIPT M15_HEAP_DESCRIPTOR \
                M15_GSTACK_RECEIPT M15_GRAPH_COMMIT_RECEIPT \
                M18_REPLICATION_RECEIPT PF4H_GRAPH_OPERATOR_GATE
            ;;
        m19)
            printf '%s\n' \
                M15_SELECTION_RECEIPT M15_ACTIVATION_RECEIPT \
                M15_SYMMETRIC_OFFSET_RECEIPT M15_HEAP_DESCRIPTOR \
                M15_GSTACK_RECEIPT M15_GRAPH_COMMIT_RECEIPT \
                M18_REPLICATION_RECEIPT PF4H_GRAPH_OPERATOR_GATE
            ;;
        m20)
            printf '%s\n' \
                M15_SELECTION_RECEIPT M15_ACTIVATION_RECEIPT \
                M15_SYMMETRIC_OFFSET_RECEIPT M15_HEAP_DESCRIPTOR \
                M15_GSTACK_RECEIPT M15_GRAPH_COMMIT_RECEIPT \
                M18_REPLICATION_RECEIPT M20_CACHE_RECEIPT \
                PF4H_GRAPH_OPERATOR_GATE
            ;;
        pf4h) printf '%s\n' PF4H_ACTIVATION_RECEIPT PF4H_GRAPH_OPERATOR_GATE ;;
        *) : ;;
    esac
}

wait_ready() {
    local container="$1" arm="$2" deadline=$((SECONDS + READY_TIMEOUT_S))
    local start=$SECONDS receipt
    while ((SECONDS < deadline)); do
        if ! is_running "$container"; then
            docker logs "$container" 2>&1
            echo "ERROR: $container exited before readiness" >&2
            return 1
        fi
        if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 \
            && [[ "$(count_log "$container" "Application startup complete.")" -ge 8 ]]; then
            if docker exec "$container" test -e /tmp/vllm-pf4h-enable; then
                echo "ERROR: activation sentinel existed before activation" >&2
                return 1
            fi
            echo "$((SECONDS - start))" >"$ROOT/.startup_seconds"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $container not ready in ${READY_TIMEOUT_S}s" >&2
    docker logs "$container" 2>&1
    return 1
}

check_receipts() {
    local container="$1" arm="$2" missing=0 receipt count
    while read -r receipt; do
        if [[ -z "$receipt" ]]; then continue; fi
        count="$(count_log "$container" "$receipt")"
        echo "receipt $receipt = $count"
        if ((count < 8)); then
            echo "ERROR: $receipt seen $count times, expected >= 8" >&2
            missing=1
        fi
    done < <(required_receipts "$arm")
    if [[ "$(count_log "$container" "M15_PPERR")" -ne 0 ]]; then
        echo "ERROR: M15_PPERR present; discard this cell" >&2
        missing=1
    fi
    return "$missing"
}

# ------------------------------------------------------------------ workloads

cell_params() {
    # concurrency num_prompts input_len output_len seed
    case "$1" in
        c8)    echo "8 256 1024 512 80801" ;;
        c16)   echo "16 512 1024 512 160801" ;;
        c32)   echo "32 1024 1024 512 320801" ;;
        c512)  echo "512 2048 1024 512 5121234" ;;
        c1024) echo "1024 4096 1024 512 10241234" ;;
        # Prefill-focused headline cells: short OSL so the MoE prefill region
        # dominates and the exact-B4096 gate actually fires.
        c32p)  echo "32 1024 4096 8 320802" ;;
        c512p) echo "512 2048 4096 8 5120802" ;;
        *) echo "unknown cell $1" >&2; return 1 ;;
    esac
}

run_cell() {
    local container="$1" arm="$2" pair="$3" cell="$4" output_dir="$5"
    read -r conc prompts ilen olen seed <<<"$(cell_params "$cell")"
    docker exec "$container" python /tmp/bench_exact_token_ids.py \
        --label "${arm}_pair${pair}_${cell}" \
        --concurrency "$conc" \
        --num-prompts "$prompts" \
        --input-len "$ilen" \
        --output-len "$olen" \
        --seed "$seed" \
        --warmup-prompts "$conc" \
        --warmup-concurrency "$conc" \
        --warmup-output-len 8 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --prompt-source "$PROMPT_SOURCE" \
        --qsl-path "${QSL_PKL:+/qsl/dataset.pkl}" \
        --save-prompt-token-ids "/results/prompts_${cell}.jsonl" \
        --timeout-s 3600 \
        --output "/results/${cell}.json" \
        >"/tmp/${container}_${cell}.log" 2>&1
    docker cp "/tmp/${container}_${cell}.log" "$container:/results/${cell}_client.log"
    rm -f "/tmp/${container}_${cell}.log"
}

# Unmeasured full-size pass with a shifted prompt sample (seed +777).  Gives
# the EPLB arm real load history and rearrangements before the measured cell;
# applied identically to the non-EPLB arm so cache position stays symmetric.
run_prewarm() {
    local container="$1" arm="$2" pair="$3" cell="$4" output_dir="$5"
    read -r conc prompts ilen olen seed <<<"$(cell_params "$cell")"
    docker exec "$container" python /tmp/bench_exact_token_ids.py \
        --label "${arm}_pair${pair}_${cell}_prewarm" \
        --concurrency "$conc" \
        --num-prompts "$prompts" \
        --input-len "$ilen" \
        --output-len "$olen" \
        --seed "$((seed + 777))" \
        --warmup-prompts "$conc" \
        --warmup-concurrency "$conc" \
        --warmup-output-len 8 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --prompt-source "$PROMPT_SOURCE" \
        --qsl-path "${QSL_PKL:+/qsl/dataset.pkl}" \
        --timeout-s 3600 \
        --output "/results/${cell}_prewarm.json" \
        >"/tmp/${container}_${cell}_prewarm.log" 2>&1
    docker cp "/tmp/${container}_${cell}_prewarm.log" \
        "$container:/results/${cell}_prewarm_client.log"
    rm -f "/tmp/${container}_${cell}_prewarm.log"
}

# ------------------------------------------------------------- extra passes

profile_pass() {
    local arm="$1" output_dir="$2"
    local container="m15_prof_${RUN_TAG}_${arm}"
    mkdir -p "$output_dir/traces"
    CURRENT_CONTAINER="$container"
    launch_server "$container" "$arm" "$output_dir" 1
    wait_ready "$container" "$arm"
    docker cp "$CLIENT" "$container:/tmp/bench_exact_token_ids.py"
    # NB: 'cond && cmd' is fatal under set -e when cond is false -- the list
    # returns 1 and the driver dies silently.  Use if/fi.
    if [[ "$arm" != "stock" && "$arm" != "rccl" ]]; then
        docker exec "$container" touch /tmp/vllm-pf4h-enable
    fi
    # The /start_profile route is mounted ONLY when vLLM was launched with a
    # profiler config (vllm/entrypoints/serve/profile/api_router.py:attach_router
    # requires app.state.args.profiler_config.profiler).  VLLM_TORCH_PROFILER_DIR
    # does not exist in 0.25.1 -- it is absent from envs.py -- so setting it
    # achieves nothing and the endpoint 404s.  Probe and refuse clearly rather
    # than letting a 404 read as a driver crash.
    if ! curl -fsS -o /dev/null -X POST http://127.0.0.1:8000/start_profile; then
        echo "ERROR: /start_profile is not mounted on this server; the image" >&2
        echo "       needs --profiler-config at launch. Skipping profile pass." >&2
        stop_server "$container"
        CURRENT_CONTAINER=""
        return 1
    fi
    run_cell "$container" "$arm" 0 c32p "$output_dir"
    curl -fsS -o /dev/null -X POST http://127.0.0.1:8000/stop_profile || \
        echo "WARNING: stop_profile failed; traces may be truncated" >&2
    sleep 30
    stop_server "$container"
    CURRENT_CONTAINER=""
    ( cd "$output_dir/traces" && sha256sum ./* >SHA256SUMS ) || true
    python3 "$CAMPAIGN_SRC_DIR/summarize_m15_campaign.py" \
        --traces "$output_dir/traces" \
        --out "$output_dir/trace_summary_${arm}.json"
}

skew_pass() {
    local arm="$1" output_dir="$2"
    local container="m15_skew_${RUN_TAG}_${arm}"
    mkdir -p "$output_dir/skew"
    CURRENT_CONTAINER="$container"
    # The observer is installed IN-PROCESS in every worker, via a sitecustomize
    # on PYTHONPATH, hooking vLLM's own FusedMoERouter.select_experts.  That
    # site is arm-independent: the stock arm never loads the PF4H/M15 classes
    # but always routes through it.
    #
    # This replaces `m15_expert_skew.py --attach`, which ran as a SEPARATE
    # docker exec interpreter and monkey-patched PF4HPrepareAndFinalize in its
    # own address space -- prepare() was never called there, so its histogram
    # could only ever be zeros, and its 1800s --deadline-s made `wait` block
    # for ~30 minutes after the cell had finished.
    M15_SKEW_HOOK="$(cd "$CAMPAIGN_SRC_DIR" && pwd)/skewhook"
    mkdir -p "$M15_SKEW_HOOK"
    cp "$CAMPAIGN_SRC_DIR/m15_router_skew.py" "$M15_SKEW_HOOK/sitecustomize.py"
    export M15_SKEW_HOOK
    launch_server "$container" "$arm" "$output_dir" 0
    wait_ready "$container" "$arm"
    docker cp "$CLIENT" "$container:/tmp/bench_exact_token_ids.py"
    run_cell "$container" "$arm" 0 c32p "$output_dir"
    # Workers dump their per-rank histograms on SIGTERM during this stop.
    stop_server "$container"
    unset M15_SKEW_HOOK
    CURRENT_CONTAINER=""
}

write_bundle() {
    local bundle="$ROOT/bundle"
    mkdir -p "$bundle/prompts" "$bundle/server"
    find "$ROOT" -name 'prompts_*.jsonl' -exec cp {} "$bundle/prompts/" \;
    cp "$0" "$bundle/driver.sh"
    {
        echo "ARMS=$ARMS"
        echo "CELLS=$CELLS"
        echo "PAIRS=$PAIRS"
        echo "MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"
        echo "invocation: $0 --arms $ARMS --cells $CELLS --pairs $PAIRS"
    } >"$bundle/driver.txt"
    docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' \
        >"$bundle/server/image_digest.json" 2>/dev/null || true
    cp "$M15_SOURCES/STAGE_MANIFEST.json" "$bundle/server/" 2>/dev/null || true
    find "$ROOT" -name 'M15_INSTALL_RECEIPT.json' \
        -exec cp {} "$bundle/server/" \; 2>/dev/null || true
    ( cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 \
        | xargs -0 sha256sum >SHA256SUMS )
    python3 "$CAMPAIGN_SRC_DIR/summarize_m15_campaign.py" \
        --campaign "$ROOT" --manifest "$bundle/MANIFEST.json"
}

# ----------------------------------------------------------------- main loop

main() {
    # Fail fast, not mid-pair: arm_env runs inside a process substitution,
    # where a ${VAR:?} refusal is invisible to set -e (mapfile would read
    # zero env lines and launch an arm-less server).
    if [[ ",$ARMS," == *",m20,"* && -z "${M20_BUDGET_LAYERS:-}" ]]; then
        echo "ERROR: the m20 arm requires M20_BUDGET_LAYERS (cached model-layer csv)" >&2
        exit 2
    fi
    claim_gpus
    mkdir -p "$ROOT"
    IFS=',' read -r -a arm_list <<<"$ARMS"
    IFS=',' read -r -a cell_list <<<"$CELLS"

    for ((pair = 1; pair <= PAIRS; pair++)); do
        local -a order=("${arm_list[@]}")
        if ((pair % 2 == 0)); then
            # reverse the arm order on even pairs
            local -a reversed=()
            for ((i = ${#order[@]} - 1; i >= 0; i--)); do
                reversed+=("${order[i]}")
            done
            order=("${reversed[@]}")
        fi
        local position=0
        for arm in "${order[@]}"; do
            position=$((position + 1))
            local tag
            printf -v tag '%02d' "$pair"
            local container="m15_${RUN_TAG}_p${tag}_${position}_${arm}"
            local output_dir="$ROOT/pair_${tag}/${position}_${arm}"
            mkdir -p "$output_dir"
            docker inspect "$container" >/dev/null 2>&1 && {
                echo "ERROR: refusing name collision $container" >&2
                return 1
            }
            # Uniform pre-arm cooldown: position-2 arms measured ~25%
            # slower than position-1 (thermal/host-state carryover);
            # equalize conditions across positions.
            sleep "${ARM_COOLDOWN:-240}"
            CURRENT_CONTAINER="$container"
            launch_server "$container" "$arm" "$output_dir" 0
            wait_ready "$container" "$arm"
            cp "$ROOT/.startup_seconds" "$output_dir/startup_seconds.txt" || true
            docker cp "$CLIENT" "$container:/tmp/bench_exact_token_ids.py"
            # Fatal under set -e for the stock arm, where the test is
            # false: the list returns 1 and the driver exits right after
            # readiness, taking the server with it.
            local barm
            barm="$(base_arm "$arm")"
            if [[ "$barm" == "m15" || "$barm" == "m18" || "$barm" == "m19" || "$barm" == "m20" || "$barm" == "pf4h" ]]; then
                docker exec "$container" touch /tmp/vllm-pf4h-enable
            fi
            if ((EPLB_PREWARM)); then
                run_prewarm "$container" "$arm" "$pair" c32p "$output_dir"
            fi
            local -a cells=("${cell_list[@]}")
            if ((pair % 2 == 0)); then
                local -a rc=()
                for ((i = ${#cells[@]} - 1; i >= 0; i--)); do rc+=("${cells[i]}"); done
                cells=("${rc[@]}")
            fi
            for cell in "${cells[@]}"; do
                run_cell "$container" "$arm" "$pair" "$cell" "$output_dir"
            done
            check_receipts "$container" "$barm" | tee "$output_dir/receipts.txt"
            docker logs "$container" >"$output_dir/server.log" 2>&1
            grep -iE "eplb|balancedness|rearrang" "$output_dir/server.log" \
                | tail -300 >"$output_dir/eplb_lines.txt" || true
            grep "M15_COVERAGE" "$output_dir/server.log" \
                | tail -64 >"$output_dir/coverage_tail.txt" || true
            stop_server "$container"
            CURRENT_CONTAINER=""
        done
    done

    # SKEW FIRST: it is the primary diagnostic deliverable, so it must not sit
    # behind the profiler pass.  Each pass is invoked in a condition context,
    # which disables errexit inside it -- one pass failing can no longer take
    # the driver (or the other pass) down with it.
    if ((SKEW_PASS)); then
        for arm in "${arm_list[@]}"; do
            if ! skew_pass "$arm" "$ROOT/skew/${arm}"; then
                echo "WARNING: skew pass FAILED for arm $arm; continuing" >&2
                salvage_container
            fi
        done
    fi
    if ((PROFILE_PASS)); then
        for arm in "${arm_list[@]}"; do
            if ! profile_pass "$arm" "$ROOT/profile/${arm}"; then
                echo "WARNING: profile pass FAILED for arm $arm; continuing" >&2
                salvage_container
            fi
        done
    fi
    if ((MAKE_BUNDLE)); then
        write_bundle
    fi
    release_gpus
    echo "campaign complete: $ROOT"
}

main "$@"
