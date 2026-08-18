#!/usr/bin/env bash
set -euo pipefail

# M15 serving campaign driver (v5).  Contract: CAMPAIGN.md + the binding
# nightshift/BENCHMARK_PROTOCOL.md; per-arm rationale and per-flag citations
# in ARMS.md next to this file.
#
# v5 = v4 plus, additively (every v4 arm still behaves exactly as it did):
#
#   1. PAIRS defaults to 6 (v5.0 said 5).  BENCHMARK_PROTOCOL.md section 3.1:
#      with a +-18% position effect and +-15% day drift, n=2 is anecdote -- and
#      v5.1's rotation balancing additionally needs a multiple of the arm
#      count, which 5 is not for 2 or 3 arms.
#   2. Four separate NATIVE arms instead of one ambiguous "native":
#        native_mirror     - v4's old `native` arm, RENAMED.  Untouched image
#                            but OUR serve line.  It is a config-matched
#                            control, NOT production; the rename exists so
#                            nobody can quote it as production again.
#        native_default    - untouched image, no PF4H env, no patch chain, and
#                            a MINIMAL serve line: shipped defaults for every
#                            knob we tuned for ourselves.
#        native_tuned_tp   - untouched image at AMD's documented recommended
#                            config for <=128 concurrency (TP8 + EP).
#        native_tuned_dp   - untouched image at AMD's documented recommended
#                            config for >=512 concurrency (DP8 + EP).
#      Fairness checklist item 2 ("baseline not sandbagged") is why the tuned
#      arms exist: beating a lazy default is not beating production.
#   3. c1det, a deterministic accuracy cell (conc 1, prefix caching and
#      chunked prefill OFF) whose output-stream SHA must match across arms.
#      The perf cells CANNOT be made deterministic -- continuous batching plus
#      chunked prefill plus DP combine ordering make stock-vs-stock SHAs
#      differ (measured: 4ee5ba42 vs 936b4f5a) -- so accuracy gets its own
#      launch variant instead of an unresolved caveat.
#   4. Open-loop cells o50p/o75p/o90p driven by bench_exact_token_ids_v3.py's
#      arrival driver at a fraction of a MEASURED closed-loop ceiling
#      (OPEN_LOOP_BASE_RATE, required; no default, because a guessed
#      saturation point silently decides the answer).
#   5. A config-dump receipt per arm (image digest, container env, resolved
#      server config, patch-marker census) so the fairness audit can check
#      baseline authenticity from EVIDENCE rather than from intent.
#
# v4 = v3 plus the true-production `native` arm.
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
# Arm order ROTATES across pairs (and reverses every k pairs) and workload
# order alternates, so neither arm nor workload can inherit a warm cache
# position.  See main(): with k arms, PAIRS must be a multiple of k or the
# balance is only nominal.
#
# --- v5.1 (2026-08-18): adversarial-review fixes ------------------------------
# Every item below closed a BLOCKING or MAJOR review finding.  They are listed
# here because each one changes what the campaign MEASURES, not just how it
# runs, and a reader comparing against a v5.0 run must know which.
#
#   F1  Order balancing was list reversal only, so with 3 arms the middle arm
#       never moved and an odd PAIRS gave arm 1 an extra position-1 slot.  Now
#       a cyclic ROTATION (+ reversal every k pairs) with a hard preflight that
#       PAIRS % n_arms == 0.  The full +-18-25% position effect used to land on
#       the headline ratio.
#   F2  Prefix caching was ON for our arms and OFF for the tuned native arms,
#       while the QSL generator emitted ~11% exact duplicate prompts per cell.
#       Prefix caching is now OFF for EVERY arm by default (PREFIX_CACHING=1
#       restores it, loudly and symmetrically), and the QSL generator strides
#       the pool so a cell contains no exact duplicates at all.
#   F3  PROMPT_SOURCE defaulted to synthetic and no invocation set it, so a
#       whole night could run uniform-random tokens and produce a headline with
#       no routing skew.  Synthetic now requires ALLOW_SYNTHETIC_PROMPTS=1.
#   F4  The determinism launch widened max_num_batched_tokens to 32768, moving
#       the mega off its B4096 operating point; native_default never got the
#       flag at all and could not start.  Now every det arm runs
#       DET_MAX_MODEL_LEN == DET_MAX_BATCHED_TOKENS == 4608, which satisfies
#       vLLM's constraint AND keeps a 4096-token prefill step.
#   F5  The accuracy gate demanded a byte-identical SHA across arms that are
#       numerically different by construction (fp8 KV vs bf16, DP8 vs TP8,
#       different attention backend), so it could only ever FAIL.  It is now a
#       CLASS gate: exact equality inside a numerics class (m15 vs stock), a
#       measured per-prompt agreement fraction across classes, a missing arm is
#       a FAIL rather than a skip, and the candidate's megakernel execution is
#       asserted so a PASS cannot be produced by an inert kernel.
#   F6  c1det at concurrency 1 under DP8 leaves 7 ranks empty, so the ragged
#       seal cannot go unanimous and the gate measured stock-vs-stock.  The
#       accuracy pass now also runs c8det (conc 8) on the same det server, and
#       the gate records whether the mega actually executed.
#   F7  native_mirror kept our serve line (MoRI all2all) but lost MORI_GPU_ARCHS
#       / MORI_SHMEM_MODE / AITER.  It now gets our RUNTIME env (minus every
#       VLLM_PF4H_* variable, which authenticity still forbids) so it controls
#       for the patch chain and nothing else.
#   F8  A rejected startup flag on any arm killed the whole driver mid-matrix.
#       An arm failure now voids its PAIR (PAIR_VOID.txt, which the analyzer
#       drops) and the campaign continues with the remaining balanced pairs.
#   F9  A receipt shortfall was fatal (pipefail) rather than discarding the
#       cell.  It now writes RECEIPT_GATE=FAIL, which the analyzer treats as
#       VOID, and the driver continues.
#   F10 B0/B1/B2 shared one RUN_TAG/ROOT, so the third invocation either hit
#       the name-collision refusal or merged into the first one's pairs.  ROOT
#       reuse with a different invocation spec is now refused, stale STOPPED
#       containers are removed, and a pair directory that already holds cell
#       manifests is refused.
#   F11 OPEN_LOOP_BASE_RATE was taken from the candidate's own position-1 run
#       and then offered to every arm.  It is now documented and enforced as
#       the MINIMUM measured ceiling across the arms in the invocation, and
#       OPEN_LOOP_BASE_RATE_SOURCE (free text, recorded) is required.
#   F12 Cell names were only validated when run_cell reached them, hours in.
#       Preflight now resolves every cell (and every open-loop rate).

readonly RUN_TAG="${RUN_TAG:-m15r1}"
readonly DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
readonly ROOT="${ROOT:-/home/subvadla/${DATE_TAG}_m15_campaign_${RUN_TAG}}"
readonly PACKET="${PACKET:?set PACKET to the deployed amd-master packet checkout}"
readonly N2="${N2:?set N2 to the exact k0_n2*.so}"
readonly M15_SOURCES="${M15_SOURCES:?set M15_SOURCES to the staged deploy dir}"
readonly DHK_ROOT="${DHK_ROOT:?set DHK_ROOT to the Distributed-HipKittens checkout}"
readonly CLIENT="${CLIENT:?set CLIENT to bench_exact_token_ids_v3.py}"
# v5 needs the v3 client: the open-loop cells and the ITL percentiles do not
# exist in v2, and a v2 client would run an "o75p" cell closed-loop while
# labelling it open.  Refuse at parse time, not 40 minutes in.
if ! grep -q '_run_open' "$CLIENT"; then
    echo "ERROR: CLIENT=$CLIENT has no _run_open; v5 requires" >&2
    echo "       bench_exact_token_ids_v3.py (open-loop + ITL)." >&2
    exit 2
fi
readonly MODEL="${MODEL:-/hf_home/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52}"
readonly IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.25.1}"
readonly READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
readonly STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-180}"
readonly MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
# Real-text prompts. "qsl" replays the official MLPerf DeepSeek-R1 set so
# expert routing carries production-like popularity skew; "synthetic" is the
# original uniform-random token stream.
#
# F3: the default stays "synthetic" for backward compatibility with the v2/v4
# recipes, but main()'s preflight REFUSES to run it without an explicit
# ALLOW_SYNTHETIC_PROMPTS=1.  BENCHMARK_PROTOCOL.md section 2 forbids synthetic
# tokens for a headline: uniform-random ids give expert routing no popularity
# skew, which is the single variable the whole MoE claim rests on, and nothing
# downstream could tell the difference after the fact.
readonly PROMPT_SOURCE="${PROMPT_SOURCE:-synthetic}"
readonly QSL_PKL="${QSL_PKL:-}"
readonly ALLOW_SYNTHETIC_PROMPTS="${ALLOW_SYNTHETIC_PROMPTS:-0}"
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

# --- v5 additions ------------------------------------------------------------
# Every native_* arm runs the untouched image: no PF4H env, no apply.py, no
# coverage/M23/RR patch, no PF4H mounts at all.  base_arm() leaves these names
# alone (they carry neither the _eplb nor the _rr suffix), so this predicate is
# the single place that decides "is this an untouched-image arm".
arm_is_native() { [[ "$(base_arm "$1")" == native_* ]]; }
# The tuned native arms are the AMD-documented configs; native_default is the
# shipped-defaults arm; native_mirror is our-config-on-untouched-image.
arm_is_native_tuned() {
    local a; a="$(base_arm "$1")"
    [[ "$a" == "native_tuned_tp" || "$a" == "native_tuned_dp" ]]
}
# F7.  Two DIFFERENT questions were being answered by one predicate:
#   (a) "does this arm run the untouched image?"  -> arm_is_native: no
#       apply.py, no coverage/M23/RR patch, no PF4H bind mounts, no VLLM_PF4H_*.
#   (b) "is this arm the VENDOR's configuration?" -> arm_is_vendor_config:
#       additionally no runtime environment of ours at all.
# native_mirror is (a) but NOT (b): it is our serve line on an untouched image,
# which is what makes it the integration-effect control.  v5.0 withheld the
# whole common env from it, including MORI_GPU_ARCHS/MORI_SHMEM_MODE, while
# still serving --all2all-backend mori_high_throughput and while disabling
# AITER -- so it either hung at 8-rank MoRI init or measured a configuration
# nobody runs, and either way it was not a control for "the patch chain".
# It now receives our RUNTIME env minus every VLLM_PF4H_* variable (which the
# authenticity gate in capture_config_receipt still forbids for any native arm).
arm_is_vendor_config() {
    local a; a="$(base_arm "$1")"
    [[ "$a" == native_* && "$a" != "native_mirror" ]]
}

# F5.  The NUMERICS CLASS of an arm: the set of serve-line properties that
# decide whether two arms can even in principle produce a bit-identical output
# token stream -- KV cache dtype, parallelism layout, attention backend, and
# the all2all backend (which fixes the combine reduction order).  Arms in the
# SAME class must agree token-for-token; arms in DIFFERENT classes are not
# bit-comparable by construction and demanding it of them produces a FAIL that
# says nothing about correctness.  v5.0's gate compared every arm as one set,
# so with {m15, native_tuned_tp} it could only ever FAIL and would then void
# the night for a reason that was baked in before the first token.
numerics_class() {
    case "$(base_arm "$1")" in
        rccl)            echo "dp8-fp8kv-rccl" ;;
        native_default)  echo "tp8-defaultkv-defaultattn" ;;
        native_tuned_tp) echo "tp8-defaultkv-aitermla" ;;
        native_tuned_dp) echo "dp8-defaultkv-aitermla" ;;
        # stock/pf4h/m15/m18/m19/m20/native_mirror all serve the SAME line:
        # DP8, fp8 KV, default attention backend, MoRI all2all.  m15 vs stock
        # is therefore a real bit-exactness question and the one the kernel
        # claim actually depends on.
        *)               echo "dp8-fp8kv-mori" ;;
    esac
}

# Open-loop cells refuse to guess.  OPEN_LOOP_BASE_RATE is the MEASURED
# closed-loop request-rate ceiling in req/s (completed requests / wall from the
# saturated c32p cell of the SAME arm family); o50p/o75p/o90p are 50/75/90% of
# it.  Leaving it unset is a hard failure the moment an o-cell is selected --
# picking a number here would quietly decide whether the pad/dummy-skip win is
# real or a benchmark artifact, which is the whole question.
#
# F11.  The base rate must ALSO be unbiased across arms.  Taking the
# candidate's own ceiling and offering it to a slower baseline turns that
# baseline's queue into its latency and manufactures an open-loop "win".  The
# contract is therefore: OPEN_LOOP_BASE_RATE is the MINIMUM measured
# closed-loop ceiling across every arm in THIS invocation, so no arm is offered
# a load it cannot absorb, and every arm sees the identical workload (which is
# also the only configuration the analyzer will compare).
# OPEN_LOOP_BASE_RATE_SOURCE is free text recording where the number came from
# (which arms, which run, which files); it is required and is written into the
# campaign receipt so the audit does not have to take it on trust.
readonly OPEN_LOOP_BASE_RATE="${OPEN_LOOP_BASE_RATE:-}"
readonly OPEN_LOOP_BASE_RATE_SOURCE="${OPEN_LOOP_BASE_RATE_SOURCE:-}"
readonly OPEN_LOOP_BURSTINESS="${OPEN_LOOP_BURSTINESS:-1.0}"

# c1det (accuracy gate) needs prefix caching and chunked prefill OFF, which are
# LAUNCH flags -- so the gate gets its own short-lived server per arm rather
# than sharing the perf server.  ACCURACY_PASS=1 enables it; it runs once per
# arm after the measured pairs.
ACCURACY_PASS="${ACCURACY_PASS:-0}"
# F4.  With chunked prefill disabled vLLM requires
# max_num_batched_tokens >= max_model_len.  v5.0 satisfied that by raising the
# BATCHING WINDOW to 32768, which moved the megakernel off the B4096 operating
# point every perf cell measures -- the gate then certified a code path the
# claim is not about.  v5.1 instead LOWERS max_model_len to the smallest value
# c1det/c8det need (ISL 4096 + OSL 8, rounded up to 4608) and sets the batching
# window to the same number.  The constraint holds, a single 4096-token prefill
# step still fills the mega's (512,4096] ragged-seal bucket, and both values
# are identical across every arm so neither side is tilted.
#
# It also has to be emitted by EVERY arm, including native_default: an arm that
# turns chunked prefill off without widening the window relative to
# max_model_len simply refuses to start (v5.0's native_default did exactly
# that, and the failure was downgraded to a WARNING).
readonly DET_MAX_BATCHED_TOKENS="${DET_MAX_BATCHED_TOKENS:-4608}"
readonly DET_MAX_MODEL_LEN="${DET_MAX_MODEL_LEN:-4608}"
# F2.  Prefix caching is a CROSS-ARM WORKLOAD knob, not a per-arm tuning knob:
# with it on for one arm and off for another, the two arms do different amounts
# of prefill on the same prompt list.  The QSL generator's stride fix removes
# exact duplicates inside a cell, but near-duplicate document prefixes remain
# possible, so the campaign closes the hole from both sides and serves every
# arm with prefix caching OFF.  PREFIX_CACHING=1 restores it -- symmetrically,
# for all arms, and it is recorded in every serve_cmd.txt.
readonly PREFIX_CACHING="${PREFIX_CACHING:-0}"
# Shared capacity guard.  Applied IDENTICALLY to every arm including
# native_default: it is a memory ceiling, not a throughput tuning knob, and the
# model's own config default (163840) does not fit the KV cache at any
# gpu-memory-utilization we can run.  Documented in ARMS.md as the single
# non-default flag the shipped-defaults arm carries.
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# AMD's DP+EP recommendation says nothing about the API front end.  We give the
# DP arm the same 8 API servers our arm uses so the front end cannot be the
# thing that loses at concurrency 512 -- documented deviation, anti-sandbagging
# (fairness item 2), overridable to 1 to measure it.
readonly NATIVE_DP_API_SERVERS="${NATIVE_DP_API_SERVERS:-8}"
readonly NATIVE_TUNED_BATCHED_TOKENS="${NATIVE_TUNED_BATCHED_TOKENS:-16384}"
# -----------------------------------------------------------------------------

PROFILE_PASS=0
SKEW_PASS=0
MAKE_BUNDLE=0
# F1: 6, not 5.  BENCHMARK_PROTOCOL.md section 3.1 makes >=5 pairs binding, and
# the rotation needs a multiple of the arm count -- 6 is the smallest number
# that is both >=5 and divisible by 2 and by 3, i.e. by every arm count these
# campaigns actually run.
PAIRS="${PAIRS:-6}"
ARMS="${ARMS:-stock,m15}"
CELLS="${CELLS:-c16,c32,c512,c1024}"
CURRENT_CONTAINER=""

usage() {
    cat <<'USAGE'
run_m15_campaign_eplb_v5.sh [options]
  --arms a,b        arms to run: stock, pf4h, m15, m18, m19, m20, rccl,
                    native_default, native_tuned_tp, native_tuned_dp,
                    native_mirror                              (default stock,m15)
                    native_default  = untouched image, shipped defaults
                                      (THE production headline baseline)
                    native_tuned_tp = untouched image, AMD <=128-conc config
                    native_tuned_dp = untouched image, AMD >=512-conc config
                    native_mirror   = untouched image, OUR serve line
                                      (control only -- never quote as production)
                    suffix _eplb = the same arm with vLLM EPLB enabled
                    suffix _rr   = the same arm with round-robin expert placement
                                   (VLLM_PF4H_RR_PLACEMENT=1 +
                                    --expert-placement-strategy round_robin;
                                    needs RR_PATCH=/path/to/rr_patch.py)
                    m18 = m15 + static hot-expert replication (M18_REP_EXPERTS env overrides the measured top-16 set)
                    m20 = m19 routing + budgeted replica cache (M20_BUDGET_LAYERS = cached model-layer csv, required)
  --cells c,...     c8 c16 c32 c512 c1024 c32p c512p           (default c16,c32,c512,c1024)
                    o50p o75p o90p = open-loop c32p traffic at 50/75/90% of
                    OPEN_LOOP_BASE_RATE (env var, req/s, REQUIRED for these)
                    c1det = the deterministic accuracy cell (see --accuracy-pass)
                    c1det/c8det = determinism cells; NOT selectable here,
                    they need --accuracy-pass
  --pairs N         order-balanced pairs.  MUST be a multiple of the number of
                    arms: the balancing is a rotation, and a non-multiple hands
                    one arm extra position-1 slots (the +-18% position effect
                    is the size of the whole claim).             (default 6)
  --accuracy-pass   after the pairs, run c1det AND c8det once per arm on a
                    determinism launch (no prefix cache, no chunked prefill,
                    max_model_len 4608) and run the class gate: exact SHA
                    equality inside a numerics class, measured per-prompt
                    agreement across classes, missing arm = FAIL, candidate
                    megakernel liveness asserted
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
        --accuracy-pass) ACCURACY_PASS=1; shift ;;
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

# F10.  v5.0 refused any name collision and never removed a container, so the
# second invocation that reused RUN_TAG died on a STOPPED leftover whose logs
# had already been harvested into server.log -- and an operator clearing it by
# hand would then have merged the two runs' pair directories.  Remove a stopped
# leftover (its evidence is already on disk); refuse a RUNNING one, because
# that is a real conflict.
reserve_container_name() {
    local container="$1"
    if docker inspect "$container" >/dev/null 2>&1; then
        if is_running "$container"; then
            echo "ERROR: refusing name collision: $container is RUNNING" >&2
            echo "       another campaign owns this node; set RUN_TAG or stop it" >&2
            return 1
        fi
        echo "NOTE: removing stopped leftover container $container" >&2
        docker rm "$container" >/dev/null 2>&1 || true
    fi
    return 0
}

# F10.  A pair directory that already holds a cell manifest belongs to an
# EARLIER invocation.  Writing into it silently promotes that run's calibration
# arm into this run's pair 1, which the integrity table would then bless.
guard_output_dir() {
    local output_dir="$1" existing
    existing="$(find "$output_dir" -maxdepth 1 -name '*.json' \
        ! -name 'docker_*.json' -print -quit 2>/dev/null || true)"
    if [[ -n "$existing" ]]; then
        echo "ERROR: $output_dir already contains results ($existing)." >&2
        echo "       This ROOT/RUN_TAG was used by an earlier invocation." >&2
        echo "       Set RUN_TAG (e.g. RUN_TAG=${RUN_TAG}b2) for a new run." >&2
        return 1
    fi
    return 0
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
        native_default)
            # SHIPPED DEFAULTS.  Deliberately EMPTY: this arm gets no env at
            # all beyond what the vendor bakes into the image, and
            # launch_server withholds the common PF4H/MoRI/HSA block from
            # every native_* arm.  Anything we add here is a knob we chose,
            # and this arm exists precisely to be the config we did not
            # choose.  Fairness item 1 (baseline authenticity).
            :
            ;;
        native_mirror)
            # v4's `native`, RENAMED.  Untouched image but OUR serve line and
            # OUR all2all backend, so it isolates "image + patch chain" from
            # "serve config".  A control, never the headline: quoting this as
            # production is exactly the past sin the rename prevents.
            printf '%s\n' \
                "-e" "VLLM_ALL2ALL_BACKEND=mori_high_throughput"
            ;;
        native_tuned_tp)
            # AMD's documented env block for vLLM on ROCm (ai-ecosystem
            # vLLM-V1 optimization guide + the 2026-02-27 vLLM/AMD blog's
            # DeepSeek-R1-0528 command).  See ARMS.md for the per-variable
            # citation.  NCCL_MIN_NCHANNELS=112 is deliberately ABSENT: AMD
            # documents it for MI300X/MI325X, not gfx950.
            printf '%s\n' \
                "-e" "VLLM_ROCM_USE_AITER=1" \
                "-e" "VLLM_ROCM_USE_AITER_MOE=1" \
                "-e" "SAFETENSORS_FAST_GPU=1" \
                "-e" "TORCH_BLAS_PREFER_HIPBLASLT=1" \
                "-e" "HIP_FORCE_DEV_KERNARG=1" \
                "-e" "VLLM_RPC_TIMEOUT=1800000"
            ;;
        native_tuned_dp)
            # Same env block plus AMD's documented all2all backend for the
            # single-node DP+EP path.  MoRI is documented by AMD as the
            # MULTI-node path, so using it here would be our choice, not
            # theirs.
            printf '%s\n' \
                "-e" "VLLM_ROCM_USE_AITER=1" \
                "-e" "VLLM_ROCM_USE_AITER_MOE=1" \
                "-e" "SAFETENSORS_FAST_GPU=1" \
                "-e" "TORCH_BLAS_PREFER_HIPBLASLT=1" \
                "-e" "HIP_FORCE_DEV_KERNARG=1" \
                "-e" "VLLM_RPC_TIMEOUT=1800000" \
                "-e" "VLLM_ALL2ALL_BACKEND=allgather_reducescatter"
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
    if [[ "$arm" == native_* ]]; then
        printf 'true'   # untouched-image arms: no install, ever
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

# ------------------------------------------------------------- serve lines
# ONE function owns every flag any arm serves with, so a reviewer can diff the
# arms by reading a single case statement instead of reconstructing a shell
# expansion.  $2 is the determinism variant (c1det): prefix caching and chunked
# prefill off, batching window widened to satisfy vLLM's
# max_num_batched_tokens >= max_model_len requirement when chunked prefill is
# disabled.
serve_flags() {
    local arm det="${2:-0}"
    arm="$(base_arm "$1")"
    # F2: prefix caching is symmetric across arms by construction.  The flag is
    # spelled out (rather than omitted) even when it matches the image default,
    # so serve_cmd.txt is a receipt a reviewer can diff without knowing what
    # 0.25.1's default happens to be.
    local prefix_flag="--no-enable-prefix-caching"
    if ((PREFIX_CACHING)); then
        prefix_flag="--enable-prefix-caching"
    fi
    local cache_flags="$prefix_flag --enable-chunked-prefill"
    local ours_batched="${MAX_BATCHED_TOKENS}"
    local tuned_batched="${NATIVE_TUNED_BATCHED_TOKENS}"
    local model_len="${MAX_MODEL_LEN}"
    if ((det)); then
        # F4: chunked prefill OFF requires max_num_batched_tokens >=
        # max_model_len.  Satisfy it by SHRINKING max_model_len to what the
        # determinism cells need rather than by inflating the batching window,
        # so the megakernel still sees a 4096-token prefill step.  Identical on
        # every arm, so it cannot tilt the gate.
        cache_flags="--no-enable-prefix-caching --no-enable-chunked-prefill"
        ours_batched="${DET_MAX_BATCHED_TOKENS}"
        tuned_batched="${DET_MAX_BATCHED_TOKENS}"
        model_len="${DET_MAX_MODEL_LEN}"
    fi
    case "$arm" in
        native_default)
            # SHIPPED DEFAULTS.  Everything absent here is absent on purpose:
            # no --gpu-memory-utilization, no --max-num-seqs, no
            # --optimization-level/--performance-mode, no chunked-prefill
            # override, no --kv-cache-dtype.
            # What remains is (a) plumbing the client needs to talk to it,
            # (b) --trust-remote-code, without which this model does not load
            # at all, (c) --tensor-parallel-size 8, without which it does not
            # fit, (d) --max-model-len, the shared capacity guard applied
            # identically to every arm, (e) --seed 0, applied identically to
            # every arm, (f) F2's symmetric prefix-caching setting, which is a
            # property of the WORKLOAD (see ARMS.md 6.7): with it on here and
            # off on the tuned arms the two baselines would not even be doing
            # the same amount of prefill on the same prompt list.
            printf '%s' "--served-model-name r1 \
               --host 0.0.0.0 --port 8000 \
               --trust-remote-code \
               --tensor-parallel-size 8 \
               --max-model-len ${model_len} \
               ${prefix_flag} \
               --seed 0"
            if ((det)); then
                # F4: this arm names no batching flag of its own, so v5.0's
                # det branch left max_num_batched_tokens at the image default
                # (8192) against max_model_len 32768 and the server refused to
                # start.  Emit BOTH, at the same values every other arm gets.
                printf '%s' " --no-enable-chunked-prefill --max-num-batched-tokens ${DET_MAX_BATCHED_TOKENS}"
            fi
            ;;
        native_tuned_tp)
            # AMD's recommended <=128-concurrency deployment.  Every flag is
            # cited in ARMS.md; none of them is ours.
            printf '%s' "--served-model-name r1 \
               --host 0.0.0.0 --port 8000 \
               --trust-remote-code \
               --tensor-parallel-size 8 \
               --enable-expert-parallel \
               --attention-backend ROCM_AITER_MLA \
               --max-model-len ${model_len} \
               --max-num-batched-tokens ${tuned_batched} \
               --max-num-seqs 2048 \
               --gpu-memory-utilization 0.9 \
               ${prefix_flag} \
               --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
               --async-scheduling \
               --seed 0"
            if ((det)); then
                printf '%s' " --no-enable-chunked-prefill"
            fi
            ;;
        native_tuned_dp)
            # AMD's recommended >=512-concurrency deployment.  --api-server-count
            # is the one documented deviation (ARMS.md): it keeps the HTTP front
            # end from becoming the bottleneck at concurrency 512, which would
            # sandbag the baseline rather than the model.
            printf '%s' "--served-model-name r1 \
               --host 0.0.0.0 --port 8000 \
               --trust-remote-code \
               --tensor-parallel-size 1 \
               --data-parallel-size 8 \
               --api-server-count ${NATIVE_DP_API_SERVERS} \
               --enable-expert-parallel \
               --disable-nccl-for-dp-synchronization \
               --attention-backend ROCM_AITER_MLA \
               --max-model-len ${model_len} \
               --max-num-batched-tokens ${tuned_batched} \
               --max-num-seqs 2048 \
               --gpu-memory-utilization 0.9 \
               ${prefix_flag} \
               --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}' \
               --async-scheduling \
               --seed 0"
            if ((det)); then
                printf '%s' " --no-enable-chunked-prefill"
            fi
            ;;
        *)
            # v4's serve line, byte-identical for stock/pf4h/m15/m18/m19/m20/
            # rccl/native_mirror.  native_mirror lands here on purpose: it IS
            # our config, which is what makes it a control and not a baseline.
            printf '%s' "--served-model-name r1 \
               --host 0.0.0.0 --port 8000 \
               --tensor-parallel-size 1 \
               --data-parallel-size 8 \
               --api-server-count 8 \
               --enable-expert-parallel \
               --all2all-backend \"\${VLLM_ALL2ALL_BACKEND}\" \
               --kv-cache-dtype fp8 \
               --gpu-memory-utilization 0.70 \
               --max-model-len ${model_len} \
               --max-num-batched-tokens ${ours_batched} \
               --max-num-seqs 128 \
               --scheduling-policy fcfs \
               ${cache_flags} \
               --optimization-level 2 \
               --performance-mode balanced \
               --seed 0"
            ;;
    esac
}

launch_server() {
    local container="$1" arm="$2" output_dir="$3" profile="${4:-0}" det="${5:-0}"
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
    if arm_is_native "$arm"; then
        # untouched-image arms: pristine image, zero patch-chain steps
        patch_chain=""
    fi
    local m23_chain=""
    if [[ -n "$M23_PATCH" && -z "$patch_chain" ]]; then
        : # native_* arm: no patch chain at all
    elif [[ -n "$M23_PATCH" ]]; then
        m23_chain="python /covpatch/m23_patch.py && "
    fi
    local rr_chain=""
    if [[ -n "$RR_PATCH" && -n "$patch_chain" ]]; then
        rr_chain="python /covpatch/rr_patch.py && "
    fi

    # The PF4H common env block and the PF4H bind mounts are OURS, not the
    # vendor's: MORI_SHMEM_MODE, HSA_*, VLLM_ROCM_USE_AITER*, /packet, /m15src,
    # /dhk, /n2, /covpatch.  A "shipped defaults" arm that inherits them is not
    # shipped defaults, so the native_* arms get NONE of them and their entire
    # environment is whatever arm_env() prints (empty for native_default) plus
    # the image's own.  Fairness item 1: the audit reads `docker inspect` and
    # must find nothing of ours there.
    #
    # F7: the RUNTIME half of that block (AITER, MoRI, HSA, HIP) is what makes
    # our serve line work at all -- MORI_GPU_ARCHS/MORI_SHMEM_MODE in
    # particular, without which an 8-rank mori_high_throughput init either
    # builds for the wrong arch or hangs.  native_mirror serves OUR line, so it
    # gets that runtime half; it does NOT get VLLM_PF4H_ACTIVATION_FILE, which
    # is a patch-chain variable and which the authenticity gate below forbids
    # in any native arm's container env.  The vendor-config arms
    # (native_default, native_tuned_*) still get nothing of ours.
    local -a common_env=()
    local -a pf4h_mounts=()
    if arm_is_native "$arm" && ! arm_is_vendor_config "$arm"; then
        common_env=(
            -e VLLM_ROCM_USE_AITER=1
            -e VLLM_ROCM_USE_AITER_MOE=1
            -e MORI_SHMEM_MODE=ISOLATION
            -e MORI_GPU_ARCHS=gfx950
            -e HIP_FORCE_DEV_KERNARG=1
            -e HSA_ENABLE_IPC_MODE_LEGACY=1
            -e HSA_NO_SCRATCH_RECLAIM=1
            -e PYTORCH_NVML_BASED_CUDA_CHECK=1
        )
    elif ! arm_is_native "$arm"; then
        common_env=(
            -e VLLM_PF4H_ACTIVATION_FILE=/tmp/vllm-pf4h-enable
            -e VLLM_ROCM_USE_AITER=1
            -e VLLM_ROCM_USE_AITER_MOE=1
            -e MORI_SHMEM_MODE=ISOLATION
            -e MORI_GPU_ARCHS=gfx950
            -e HIP_FORCE_DEV_KERNARG=1
            -e HSA_ENABLE_IPC_MODE_LEGACY=1
            -e HSA_NO_SCRATCH_RECLAIM=1
            -e PYTORCH_NVML_BASED_CUDA_CHECK=1
        )
        pf4h_mounts=(
            -v "$PACKET:/packet:ro"
            -v "$M15_SOURCES:/m15src:ro"
            -v "$DHK_ROOT:/dhk:ro"
            -v "$N2:/n2/k0_n2.cpython-312-x86_64-linux-gnu.so:ro"
            -v "$COV_PATCH:/covpatch/coverage_patch.py:ro"
        )
        if [[ -n "$DHK_GITDIR" ]]; then
            pf4h_mounts+=(-v "$DHK_GITDIR:$DHK_GITDIR:ro")
        fi
        if [[ -n "$M23_PATCH" ]]; then
            pf4h_mounts+=(-v "$M23_PATCH:/covpatch/m23_patch.py:ro")
        fi
        if [[ -n "$RR_PATCH" ]]; then
            pf4h_mounts+=(-v "$RR_PATCH:/covpatch/rr_patch.py:ro")
        fi
    fi
    local -a skew_args=()
    if [[ -n "${M15_SKEW_HOOK:-}" ]]; then
        skew_args=(
            -v "$M15_SKEW_HOOK:/skewhook:ro"
            -e PYTHONPATH=/skewhook
            -e M15_SKEW_OUT=/results/skew
            -e "M15_SKEW_CAPTURE_LAYERS=${M15_SKEW_CAPTURE_LAYERS:-}"
            -e "M15_SKEW_CAPTURE_MAX=${M15_SKEW_CAPTURE_MAX:-64}"
        )
    fi
    local -a qsl_args=()
    if [[ -n "$QSL_PKL" ]]; then
        qsl_args=(-v "$QSL_PKL:/qsl/dataset.pkl:ro")
    fi

    local flags
    flags="$(serve_flags "$arm" "$det")"
    # The exact serve line is a fairness receipt in its own right: it is what a
    # reviewer diffs when asking "was the baseline given our flags?".
    printf 'arm=%s det=%s\nvllm serve %s %s%s%s%s\n' \
        "$arm" "$det" "$MODEL" "$flags" \
        "$eplb_args" "$rr_args" "$native_args" >"$output_dir/serve_cmd.txt"

    # shellcheck disable=SC2046
    docker run -d \
        --name "$container" \
        --network host --ipc host --shm-size 32g \
        --device /dev/kfd --device /dev/dri \
        --group-add video --group-add render \
        "${env_args[@]}" "${profile_args[@]}" \
        "${common_env[@]}" \
        -v /data/hf_home:/hf_home:ro \
        "${pf4h_mounts[@]}" \
        "${qsl_args[@]}" \
        "${skew_args[@]}" \
        -v "$output_dir:/results" \
        -v "$output_dir:/prof" \
        --entrypoint bash \
        "$IMAGE" \
        -lc "test ! -e /tmp/vllm-pf4h-enable && \
             $(apply_command "$arm") && \
             ${patch_chain}${m23_chain}${rr_chain}exec vllm serve '$MODEL' \
               ${flags}${eplb_args}${rr_args}${native_args}" >/dev/null
}

# ------------------------------------------------------------ config receipts
# Fairness checklist items 1 and 3 are answered by EVIDENCE, so every arm --
# ours and the vendor's alike -- deposits the same four artefacts next to its
# results.  For the native_* arms the patch-marker census is a GATE, not a
# note: a single one of our markers in an untouched-image arm means the arm is
# not what its name says and the pair is void.
capture_config_receipt() {
    local container="$1" arm="$2" output_dir="$3"
    docker image inspect "$IMAGE" \
        --format '{{json .RepoDigests}}{{"\n"}}{{.Id}}' \
        >"$output_dir/image_digest.txt" 2>/dev/null || true
    docker inspect "$container" --format '{{json .Config.Env}}' \
        >"$output_dir/docker_env.json" 2>/dev/null || true
    docker inspect "$container" --format '{{json .Config.Cmd}}' \
        >"$output_dir/docker_cmd.json" 2>/dev/null || true
    # vLLM dumps its resolved configuration once per engine at startup; keep
    # the config-ish lines so the audit can read the ACTUAL values rather than
    # trusting the flags we believe we passed (0.25.1 has moved defaults since
    # the AMD docs were written).
    docker logs "$container" 2>&1 \
        | grep -iE "Initializing|config|args|parallel|cudagraph|aiter|attention backend|prefix cach|chunked" \
        | head -200 >"$output_dir/server_config_dump.txt" || true

    local marker count total=0
    : >"$output_dir/patch_markers.txt"
    for marker in PF4H_INTEGRATION_PATCH_V3_M15 PF4H_COVERAGE_PATCH_V1 \
                  PF4H_M23_RAGGED_SEAL_V1 PF4H_RR_ VLLM_PF4H_; do
        count="$(count_log "$container" "$marker")"
        echo "$marker = $count" >>"$output_dir/patch_markers.txt"
        total=$((total + count))
    done
    if arm_is_native "$arm"; then
        # Also check the container ENV, not just the log: an inherited
        # VLLM_PF4H_* variable is exactly how "production" got faked before.
        if grep -q 'VLLM_PF4H_' "$output_dir/docker_env.json" 2>/dev/null; then
            echo "ERROR: native arm $arm has VLLM_PF4H_* in its container env" >&2
            cat "$output_dir/docker_env.json" >&2
            echo "NATIVE_AUTHENTICITY=FAIL(env)" >>"$output_dir/patch_markers.txt"
            return 1
        fi
        if ((total != 0)); then
            echo "ERROR: native arm $arm shows $total PF4H patch marker(s)" >&2
            cat "$output_dir/patch_markers.txt" >&2
            echo "NATIVE_AUTHENTICITY=FAIL(markers)" >>"$output_dir/patch_markers.txt"
            return 1
        fi
        echo "NATIVE_AUTHENTICITY=PASS" >>"$output_dir/patch_markers.txt"
    fi
    return 0
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

# How many "Application startup complete." lines mean READY.  v4 hardcoded 8
# because every v4 arm ran --api-server-count 8.  The TP8 native arms run ONE
# API server, so an unconditional 8 would hang them until READY_TIMEOUT_S and
# score production as a failure -- a fairness bug wearing a timeout costume.
expected_startups() {
    case "$(base_arm "$1")" in
        native_default|native_tuned_tp) echo 1 ;;
        native_tuned_dp)                echo "$NATIVE_DP_API_SERVERS" ;;
        *)                              echo 8 ;;
    esac
}

# $3 (optional) is the arm's output dir.  v5.0 wrote ONE shared
# $ROOT/.startup_seconds that every arm overwrote and only the pair loop ever
# copied, so the accuracy/profile/skew passes left no startup record at all and
# NIGHT_SCHEDULE's "read startup_seconds.txt" instruction named a file that did
# not exist yet.  Write it per arm, at the source.
wait_ready() {
    local container="$1" arm="$2" out="${3:-}" deadline=$((SECONDS + READY_TIMEOUT_S))
    local start=$SECONDS receipt want
    want="$(expected_startups "$arm")"
    while ((SECONDS < deadline)); do
        if ! is_running "$container"; then
            # Flag drift is the expected failure mode for the native_tuned_*
            # arms: AMD's published commands target vLLM 0.14.0rc2 and we run
            # 0.25.1, so a flag can be renamed, removed, or turned into a
            # BooleanOptionalAction.  Say so in one loud line with the log
            # tail attached, instead of leaving "exited before readiness" for
            # someone to reverse-engineer at 4am.
            local logs
            logs="$(docker logs "$container" 2>&1 || true)"
            if grep -qiE "unrecognized argument|invalid choice|no such option|unexpected keyword|error: argument|unrecognized option" <<<"$logs"; then
                echo "ERROR: STARTUP FLAG REJECTED by vLLM in arm '$arm'" >&2
                echo "       image=$IMAGE  container=$container" >&2
                echo "       the serve line is in the run's serve_cmd.txt" >&2
                grep -iE "unrecognized argument|invalid choice|no such option|unexpected keyword|error: argument|unrecognized option" <<<"$logs" \
                    | head -10 >&2
            fi
            echo "--- last 120 log lines -------------------------------" >&2
            tail -120 <<<"$logs" >&2
            echo "------------------------------------------------------" >&2
            echo "ERROR: $container exited before readiness" >&2
            return 1
        fi
        if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 \
            && [[ "$(count_log "$container" "Application startup complete.")" -ge "$want" ]]; then
            if docker exec "$container" test -e /tmp/vllm-pf4h-enable; then
                echo "ERROR: activation sentinel existed before activation" >&2
                return 1
            fi
            echo "$((SECONDS - start))" >"$ROOT/.startup_seconds"
            if [[ -n "$out" ]]; then
                echo "$((SECONDS - start))" >"$out/startup_seconds.txt"
            fi
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
    # F9 + fairness item 3: the ragged-seal COVERAGE is the only receipt that
    # says the megakernel executed the traffic rather than merely installing
    # itself.  v5.0 printed the last RAGGED_SEAL_RECEIPT line and thresholded
    # nothing, so an arm sealing 5% of steps and one sealing 99% were
    # indistinguishable.  Parse sealed/in_bucket and fail below
    # MIN_SEAL_COVERAGE_PCT (default 90).
    case "$(base_arm "$arm")" in
        m15|m18|m19|m20)
            local seal_line sealed in_bucket pct
            seal_line="$(docker logs "$container" 2>&1 \
                | grep -F 'RAGGED_SEAL_RECEIPT' | tail -1 || true)"
            if [[ -z "$seal_line" ]]; then
                echo "coverage RAGGED_SEAL_RECEIPT = absent"
                echo "ERROR: no RAGGED_SEAL_RECEIPT: the megakernel never" >&2
                echo "       reported coverage; this arm is not evidence" >&2
                missing=1
            else
                echo "coverage $seal_line"
                sealed="$(sed -n 's/.*sealed[= ]*\([0-9][0-9]*\).*/\1/p' <<<"$seal_line" | head -1)"
                in_bucket="$(sed -n 's/.*in_bucket[= ]*\([0-9][0-9]*\).*/\1/p' <<<"$seal_line" | head -1)"
                if [[ -n "$sealed" && -n "$in_bucket" && "$in_bucket" -gt 0 ]]; then
                    pct=$((100 * sealed / in_bucket))
                    echo "coverage seal_pct = $pct"
                    if ((pct < ${MIN_SEAL_COVERAGE_PCT:-90})); then
                        echo "ERROR: seal coverage ${pct}% < ${MIN_SEAL_COVERAGE_PCT:-90}%" >&2
                        missing=1
                    fi
                else
                    echo "coverage seal_pct = unparsed"
                    echo "ERROR: could not parse sealed/in_bucket from: $seal_line" >&2
                    missing=1
                fi
            fi
            ;;
    esac
    if ((missing)); then
        echo "RECEIPT_GATE=FAIL"
    else
        echo "RECEIPT_GATE=PASS"
    fi
    return "$missing"
}

# F9.  v5.0 ran `check_receipts | tee` with `set -o pipefail` in force, so a
# single short receipt count killed the driver mid-matrix and left an odd,
# position-unbalanced number of completed pairs -- the opposite of the comment's
# promise that "a missing receipt discards the cell".  Discarding is the right
# behaviour, so record the verdict where the analyzer can see it (it treats
# RECEIPT_GATE=FAIL as VOID and excludes those records from every ratio) and
# let the campaign continue.
record_receipts() {
    local container="$1" barm="$2" output_dir="$3" rc=0
    check_receipts "$container" "$barm" >"$output_dir/receipts.txt" 2>&1 || rc=$?
    cat "$output_dir/receipts.txt"
    if ((rc != 0)); then
        echo "WARNING: receipt gate FAILED for $barm; every cell of this arm" >&2
        echo "         is marked VOID and excluded from the analysis." >&2
    fi
    return 0
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
        # --- v5 --------------------------------------------------------
        # Accuracy gate.  conc 1 so there is no continuous batching, 32
        # prompts so it costs ~a minute, ISL/OSL matched to c32p so it
        # exercises the same prefill path the headline claims.  Its seed is
        # FIXED and shared by every arm; run_accuracy_pass launches it on a
        # determinism server (no prefix cache, no chunked prefill).
        c1det) echo "1 32 4096 8 424242" ;;
        # F6.  c1det alone proves nothing about the MEGAKERNEL: at concurrency
        # 1 under TP1/DP8 a single in-flight request lands on ONE data-parallel
        # rank while the other seven run dummy batches, the (512,4096] ragged
        # seal cannot go unanimous, and the m15 arm falls back to stock ops --
        # so the gate would be comparing stock against stock while wearing the
        # candidate's label.  c8det puts one real request on each of the 8 DP
        # ranks so the seal can fire, at the cost of reintroducing batching
        # nondeterminism; it is therefore reported as a per-prompt AGREEMENT
        # FRACTION inside a numerics class, never as a bit-exactness gate.
        # Both run on the same determinism server, back to back.
        c8det) echo "8 64 4096 8 424243" ;;
        # Open-loop cells: c32p's traffic delivered on a Poisson schedule at
        # 50/75/90% of the measured closed-loop ceiling.  concurrency is a
        # warmup width only here -- in-flight concurrency is an OUTPUT.
        # 512 prompts at the rates we expect (~4.7 req/s ceiling at 19.3k input
        # tok/s / 4096 ISL) is an ARRIVAL SPAN of 512/(0.50*4.71) = 217 s for
        # o50p, 145 s for o75p and 121 s for o90p, plus the drain tail and the
        # 32-prompt ISL-4096 warmup.  Budget ~300 / ~220 / ~190 s per cell --
        # NOT the "near two minutes" v5.0 claimed, which under-budgeted o50p by
        # about two minutes per arm per pair.
        o50p)  echo "32 512 4096 8 320850" ;;
        o75p)  echo "32 512 4096 8 320875" ;;
        o90p)  echo "32 512 4096 8 320890" ;;
        *) echo "unknown cell $1" >&2; return 1 ;;
    esac
}

# F12: `o75` (no trailing p) used to match this glob, pass the preflight, and
# then die inside cell_rate hours later.  The predicate now names the cells
# that exist.
cell_is_open() {
    case "$1" in
        o50p|o75p|o90p) return 0 ;;
        *) return 1 ;;
    esac
}
# Cells that only the accuracy pass may run: they need the determinism launch.
cell_is_det() { [[ "$1" == "c1det" || "$1" == "c8det" ]]; }

# Offered request rate for an open-loop cell, in req/s.  Derived from the
# MEASURED closed-loop ceiling, never guessed: OPEN_LOOP_BASE_RATE is read off
# the saturated c32p run (result.completed / result.wall_seconds).
#
# F11.  Two constraints pull against each other here:
#   * an arm offered more than it can serve reports its QUEUE as its latency,
#     so the rate must not exceed the slowest arm's ceiling; and
#   * two arms offered DIFFERENT rates are not running the same workload, so
#     the analyzer refuses to compare them (and v5.0's per-arm-family recipe
#     therefore could never produce a legitimate cross-arm o-cell at all).
# The only configuration that satisfies both is a SINGLE offered rate, equal to
# a fraction of the MINIMUM measured ceiling across the arms in this
# invocation.  That is what OPEN_LOOP_BASE_RATE must be, and
# OPEN_LOOP_BASE_RATE_SOURCE must say which arms were measured to get it.
cell_rate() {
    local cell="$1" frac
    case "$cell" in
        o50p) frac="0.50" ;;
        o75p) frac="0.75" ;;
        o90p) frac="0.90" ;;
        *) return 1 ;;
    esac
    if [[ -z "$OPEN_LOOP_BASE_RATE" ]]; then
        echo "ERROR: cell $cell is open-loop and needs OPEN_LOOP_BASE_RATE" >&2
        echo "       (req/s, measured: completed/wall_seconds from a saturated" >&2
        echo "        closed-loop c32p run of THIS arm family).  Refusing to" >&2
        echo "        guess a saturation point -- the guess would decide the" >&2
        echo "        result." >&2
        return 1
    fi
    awk -v base="$OPEN_LOOP_BASE_RATE" -v f="$frac" \
        'BEGIN { if (base+0 <= 0) exit 1; printf "%.6f", base * f }'
}

run_cell() {
    local container="$1" arm="$2" pair="$3" cell="$4" output_dir="$5"
    read -r conc prompts ilen olen seed <<<"$(cell_params "$cell")"
    # Open-loop cells add exactly two flags; every other argument, including
    # the prompt seed and therefore the exact token stream, is unchanged.  The
    # v3 client picks its driver from the presence of --request-rate, so a
    # closed cell here is byte-for-byte the v4 invocation.
    local -a open_args=()
    if cell_is_open "$cell"; then
        local rate
        rate="$(cell_rate "$cell")" || return 1
        open_args=(--request-rate "$rate" --burstiness "$OPEN_LOOP_BURSTINESS")
        echo "cell $cell: open-loop, offered ${rate} req/s " \
             "(burstiness $OPEN_LOOP_BURSTINESS, base $OPEN_LOOP_BASE_RATE)"
    fi
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
        "${open_args[@]}" \
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

# --- v5: the deterministic accuracy gate -------------------------------------
# BENCHMARK_PROTOCOL.md item 3.4: a fast wrong kernel is a zero, and item
# "known open problem": ordered_output_token_id_stream_sha256 differs even
# stock-vs-stock in the perf cells.  That divergence is REAL and expected --
# chunked prefill splits each prefill at run-dependent boundaries, continuous
# batching changes reduction composition, and DP8 combine ordering is arrival
# dependent -- so the fix is not to argue about the perf cells but to measure
# accuracy where determinism is achievable: concurrency 1, no prefix cache, no
# chunked prefill, one fixed seed, its own short-lived server per arm.
#
# PASS means every arm produced the identical ordered output-token stream.
# FAIL is not a soft warning: unresolved parity = no claim.
accuracy_pass() {
    local arm="$1" output_dir="$2"
    local container="m15_acc_${RUN_TAG}_${arm}"
    mkdir -p "$output_dir"
    reserve_container_name "$container" || return 1
    sleep "${ARM_COOLDOWN:-240}"
    CURRENT_CONTAINER="$container"
    # main() calls this function in a condition context, which disables errexit
    # for its whole body -- so every step that can fail must say so explicitly
    # or the pass would sail on past a server that never started and produce a
    # missing-result FAIL with no diagnosis attached.
    launch_server "$container" "$arm" "$output_dir" 0 1 || return 1
    wait_ready "$container" "$arm" "$output_dir" || {
        docker logs "$container" >"$output_dir/server.log" 2>&1 || true
        return 1
    }
    capture_config_receipt "$container" "$arm" "$output_dir" || return 1
    printf '%s\n' "$(numerics_class "$arm")" >"$output_dir/numerics_class.txt"
    docker cp "$CLIENT" "$container:/tmp/bench_exact_token_ids.py"
    local barm
    barm="$(base_arm "$arm")"
    if [[ "$barm" == "m15" || "$barm" == "m18" || "$barm" == "m19" \
          || "$barm" == "m20" || "$barm" == "pf4h" ]]; then
        docker exec "$container" touch /tmp/vllm-pf4h-enable
    fi
    # c1det: the strict, deterministic cell (conc 1).
    # c8det: the same prefill shape at concurrency 8, so all eight DP ranks
    # carry real tokens and the megakernel's ragged seal can actually fire.
    # Without it a PASS says nothing about the candidate (F6).
    local acc_rc=0
    run_cell "$container" "$arm" 0 c1det "$output_dir" || acc_rc=1
    run_cell "$container" "$arm" 0 c8det "$output_dir" || acc_rc=1
    record_receipts "$container" "$barm" "$output_dir"
    docker logs "$container" >"$output_dir/server.log" 2>&1
    # The accuracy servers' own coverage evidence.  v5.0 never extracted it,
    # and the analyzer only walks pair_* directories, so the question "did the
    # megakernel execute during the gate?" had no answer anywhere.
    {
        grep -F 'RAGGED_SEAL_RECEIPT' "$output_dir/server.log" | tail -5 || true
        echo "M15_ACTIVATION_RECEIPT_count = $(grep -cF 'M15_ACTIVATION_RECEIPT' "$output_dir/server.log" || true)"
        echo "M15_GRAPH_COMMIT_RECEIPT_count = $(grep -cF 'M15_GRAPH_COMMIT_RECEIPT' "$output_dir/server.log" || true)"
    } >"$output_dir/coverage_tail.txt" 2>/dev/null || true
    stop_server "$container" || true
    CURRENT_CONTAINER=""
    return "$acc_rc"
}

# Compare every arm's c1det output-stream SHA and write ONE PASS/FAIL line the
# fairness audit can cite.  Prompt SHAs are compared too: if the inputs ever
# diverge, an output match would be meaningless.
# F5/F6.  The gate the review demanded, in four parts:
#
#   1. COMPLETENESS.  Every arm the campaign was asked to run must have
#      produced BOTH determinism cells.  v5.0 skipped a missing c1det.json with
#      `continue`, required only two rows, and downgraded an accuracy-pass
#      failure to a WARNING -- so it could print ACCURACY_GATE=PASS arms=2 over
#      {m15, stock} while the production baseline never started.
#   2. WITHIN-CLASS EXACTNESS.  Arms that share a numerics class must agree
#      token-for-token; this is the real kernel-parity question (m15 vs stock)
#      and it is a hard FAIL.
#   3. CROSS-CLASS AGREEMENT.  Arms in different classes are not bit-comparable
#      by construction (fp8 KV vs bf16, DP8 vs TP8, different attention
#      backend), so they are MEASURED, not asserted: per-prompt agreement
#      fraction and the first divergent prompt index, reported for the record.
#      Demanding equality here is what made v5.0's gate unpassable.
#   4. CANDIDATE LIVENESS.  A PASS produced by an inert megakernel is the
#      project's oldest sin.  The candidate must show ragged-seal coverage on
#      the accuracy server or the gate FAILs for want of evidence.
assert_accuracy_gate() {
    local acc_root="$1" candidate="$2"; shift 2
    local out="$acc_root/ACCURACY_GATE.txt" rc=0
    # `if !` so a FAIL does not kill the driver through errexit before the
    # PASS/FAIL line has been printed where a human can see it.
    if ! python3 - "$acc_root" "$candidate" "$@" >"$out" <<'PYGATE'
import json
import os
import re
import sys

root = sys.argv[1]
candidate = sys.argv[2]
expected = list(sys.argv[3:])
CELLS = ("c1det", "c8det")

failures = []
notes = []
docs = {}      # (arm, cell) -> manifest
classes = {}   # arm -> numerics class


def read_text(path):
    try:
        with open(path) as handle:
            return handle.read()
    except (IOError, OSError):
        return ""


for arm in expected:
    arm_dir = os.path.join(root, arm)
    classes[arm] = (read_text(os.path.join(arm_dir, "numerics_class.txt")).strip()
                    or "unknown")
    for cell in CELLS:
        path = os.path.join(arm_dir, cell + ".json")
        if not os.path.isfile(path):
            failures.append(
                "missing_result arm={} cell={} (the arm never produced a "
                "determinism result; a skipped arm is not a passed arm)"
                .format(arm, cell))
            continue
        try:
            with open(path) as handle:
                docs[(arm, cell)] = json.load(handle)
        except (ValueError, IOError, OSError) as exc:
            failures.append("unreadable arm={} cell={} err={}".format(arm, cell, exc))

# ---- per-arm summary --------------------------------------------------------
print("expected_arms={}".format(",".join(expected)))
for arm in expected:
    print("arm={} numerics_class={}".format(arm, classes.get(arm, "unknown")))
for (arm, cell), doc in sorted(docs.items()):
    result = doc["result"]
    print("arm={} cell={} completed={} failed={} prompt_sha={} output_sha={}".format(
        arm, cell, result["completed"], result["failed"],
        doc["workload"]["ordered_prompt_token_id_stream_sha256"][:16],
        result["ordered_output_token_id_stream_sha256"][:16]))
    if result["failed"]:
        failures.append("failed_requests arm={} cell={} n={}".format(
            arm, cell, result["failed"]))

# ---- identical inputs -------------------------------------------------------
for cell in CELLS:
    shas = {doc["workload"]["ordered_prompt_token_id_stream_sha256"]
            for (arm, c), doc in docs.items() if c == cell}
    if len(shas) > 1:
        failures.append("prompt_sha_mismatch cell={} n_distinct={}".format(
            cell, len(shas)))


def per_prompt(doc):
    """index -> per-request output token SHA, for the requests that succeeded."""
    out = {}
    for entry in doc.get("per_query", []):
        if entry.get("ok") and entry.get("output_token_ids_sha256"):
            out[entry["index"]] = entry["output_token_ids_sha256"]
    return out


def agreement(a_doc, b_doc):
    """(matching, compared, first_divergent_index)."""
    left, right = per_prompt(a_doc), per_prompt(b_doc)
    shared = sorted(set(left) & set(right))
    match = [i for i in shared if left[i] == right[i]]
    diverged = [i for i in shared if left[i] != right[i]]
    return len(match), len(shared), (diverged[0] if diverged else None)


# ---- within-class exactness (hard) and cross-class agreement (measured) -----
by_class = {}
for arm in expected:
    by_class.setdefault(classes.get(arm, "unknown"), []).append(arm)

strict_pairs = 0
candidate_has_peer = False
for cls, arms in sorted(by_class.items()):
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            for cell in CELLS:
                if (a, cell) not in docs or (b, cell) not in docs:
                    continue
                sha_a = docs[(a, cell)]["result"]["ordered_output_token_id_stream_sha256"]
                sha_b = docs[(b, cell)]["result"]["ordered_output_token_id_stream_sha256"]
                match, compared, first = agreement(docs[(a, cell)], docs[(b, cell)])
                frac = (100.0 * match / compared) if compared else float("nan")
                print("within_class class={} {} vs {} cell={} identical={} "
                      "prompt_agreement={}/{} ({:.2f}%) first_divergent_index={}"
                      .format(cls, a, b, cell, sha_a == sha_b, match, compared,
                              frac, first))
                if candidate in (a, b):
                    candidate_has_peer = True
                if cell == "c1det":
                    strict_pairs += 1
                    if sha_a != sha_b:
                        failures.append(
                            "within_class_output_mismatch class={} {} vs {} "
                            "cell={} agreement={}/{} first_divergent_index={}"
                            .format(cls, a, b, cell, match, compared, first))
                elif sha_a != sha_b:
                    # c8det is batching-nondeterministic by construction; a
                    # mismatch here is recorded with its magnitude, not asserted.
                    notes.append(
                        "c8det divergence inside class {} ({} vs {}): {}/{} "
                        "prompts agree -- expected under continuous batching, "
                        "reported so its MAGNITUDE is on the record"
                        .format(cls, a, b, match, compared))

class_names = sorted(by_class)
for i in range(len(class_names)):
    for j in range(i + 1, len(class_names)):
        for a in by_class[class_names[i]]:
            for b in by_class[class_names[j]]:
                for cell in CELLS:
                    if (a, cell) not in docs or (b, cell) not in docs:
                        continue
                    match, compared, first = agreement(docs[(a, cell)], docs[(b, cell)])
                    frac = (100.0 * match / compared) if compared else float("nan")
                    print("cross_class {} [{}] vs {} [{}] cell={} "
                          "prompt_agreement={}/{} ({:.2f}%) "
                          "first_divergent_index={}  # NOT bit-comparable by "
                          "construction; measured, not asserted"
                          .format(a, classes[a], b, classes[b], cell,
                                  match, compared, frac, first))

# ---- candidate liveness -----------------------------------------------------
cov = read_text(os.path.join(root, candidate, "coverage_tail.txt"))
sealed_total = 0
for line in cov.splitlines():
    hit = re.search(r"sealed[= ]*([0-9]+)", line)
    if hit:
        sealed_total = max(sealed_total, int(hit.group(1)))
mega_ran = sealed_total > 0
print("candidate={} mega_executed={} max_sealed_steps={}".format(
    candidate, mega_ran, sealed_total))
if os.path.isdir(os.path.join(root, candidate)) and not mega_ran:
    failures.append(
        "candidate_inert candidate={}: no RAGGED_SEAL_RECEIPT with sealed>0 on "
        "the accuracy server, so the gate compared the FALLBACK path and says "
        "nothing about the kernel under claim".format(candidate))

for note in notes:
    print("NOTE: " + note)

if failures:
    for item in failures:
        print("FAILURE: " + item)
    print("ACCURACY_GATE=FAIL arms={} reasons={}".format(
        len(expected), len(failures)))
    sys.exit(1)
if not candidate_has_peer:
    print("ACCURACY_GATE=PASS_WEAK arms={} scope=no_within_class_peer".format(
        len(expected)))
    print("REQUIRED_CAVEAT: no arm shares the candidate's numerics class, so "
          "no bit-exactness was established for it; only cross-class "
          "agreement fractions above are evidence. Add the `stock` arm to the "
          "accuracy pass to close this.")
    sys.exit(0)
print("ACCURACY_GATE=PASS arms={} strict_comparisons={}".format(
    len(expected), strict_pairs))
PYGATE
    then
        rc=1
    fi
    cat "$out"
    return "$rc"
}

profile_pass() {
    local arm="$1" output_dir="$2"
    local container="m15_prof_${RUN_TAG}_${arm}"
    mkdir -p "$output_dir/traces"
    CURRENT_CONTAINER="$container"
    launch_server "$container" "$arm" "$output_dir" 1
    wait_ready "$container" "$arm" "$output_dir"
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
    wait_ready "$container" "$arm" "$output_dir"
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
    {
        echo "ARMS=$ARMS CELLS=$CELLS PAIRS=$PAIRS"
        echo "prompt_source=$PROMPT_SOURCE prefix_caching=$PREFIX_CACHING"
        echo
        echo "The node's summarize_m15_campaign.py reads vllm-bench field names"
        echo "(output_throughput, p50_ttft_ms) that THIS client has never"
        echo "emitted, so its numbers are structurally zero for these"
        echo "manifests.  It is invoked below only for the trace/bundle"
        echo "inventory.  The results table is produced by"
        echo "analyze_campaign_v5.py, which refuses unknown schemas instead of"
        echo "silently returning 0."
    } >"$bundle/READ_THIS_FIRST.txt"
    python3 "$CAMPAIGN_SRC_DIR/summarize_m15_campaign.py" \
        --campaign "$ROOT" --manifest "$bundle/MANIFEST.json" || {
        echo "WARNING: summarize_m15_campaign.py failed or read no rows; this" >&2
        echo "         is expected -- use analyze_campaign_v5.py." >&2
    }
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
    # F3: refuse a synthetic-prompt headline run.  Uniform-random token ids
    # give expert routing no popularity skew, which is the one variable the
    # whole MoE claim rests on, and no artefact downstream records the
    # difference in a way anyone would notice at 04:00.
    if [[ "$PROMPT_SOURCE" != "qsl" ]] && ((!ALLOW_SYNTHETIC_PROMPTS)); then
        echo "ERROR: PROMPT_SOURCE=$PROMPT_SOURCE." >&2
        echo "       BENCHMARK_PROTOCOL.md section 2: real text (MLPerf QSL)," >&2
        echo "       never synthetic tokens, for a headline -- synthetic ids" >&2
        echo "       carry no expert-routing skew." >&2
        echo "       Set PROMPT_SOURCE=qsl QSL_PKL=/path/to/dataset.pkl," >&2
        echo "       or ALLOW_SYNTHETIC_PROMPTS=1 for a deliberate" >&2
        echo "       non-headline diagnostic run." >&2
        exit 2
    fi
    if [[ "$PROMPT_SOURCE" == "qsl" && -z "$QSL_PKL" ]]; then
        echo "ERROR: PROMPT_SOURCE=qsl needs QSL_PKL=/path/to/dataset.pkl" >&2
        exit 2
    fi
    # Preflight: refuse BEFORE claiming the GPUs, not four hours in.  F12: every
    # cell name is RESOLVED here (cell_params + cell_rate), so a typo costs a
    # second rather than a pair's worth of node time.
    local c
    for c in ${CELLS//,/ }; do
        if ! cell_params "$c" >/dev/null; then
            echo "ERROR: unknown cell '$c'" >&2
            exit 2
        fi
        if cell_is_det "$c"; then
            echo "ERROR: '$c' is not a --cells cell; it needs the determinism" >&2
            echo "       launch.  Use --accuracy-pass." >&2
            exit 2
        fi
        if cell_is_open "$c"; then
            if [[ -z "$OPEN_LOOP_BASE_RATE" ]]; then
                echo "ERROR: cell '$c' is open-loop; set OPEN_LOOP_BASE_RATE (req/s)" >&2
                echo "       to the MINIMUM measured closed-loop ceiling across" >&2
                echo "       the arms in THIS invocation (completed/wall_seconds" >&2
                echo "       from each arm's saturated c32p run).  There is no" >&2
                echo "       default: guessing the saturation point decides the" >&2
                echo "       answer, and taking the candidate's own ceiling" >&2
                echo "       offers the baseline a load it cannot absorb." >&2
                exit 2
            fi
            if [[ -z "$OPEN_LOOP_BASE_RATE_SOURCE" ]]; then
                echo "ERROR: set OPEN_LOOP_BASE_RATE_SOURCE to a one-line record" >&2
                echo "       of WHICH arms were measured for the base rate and" >&2
                echo "       where those runs live.  The audit cannot verify an" >&2
                echo "       unbiased offered rate from the number alone." >&2
                exit 2
            fi
            cell_rate "$c" >/dev/null || exit 2
        fi
    done
    IFS=',' read -r -a arm_list <<<"$ARMS"
    IFS=',' read -r -a cell_list <<<"$CELLS"
    local n_arms=${#arm_list[@]}
    # F1.  POSITION BALANCE.  v5.0 reversed the arm LIST on even pairs, which
    # with 3 arms never moved the middle arm at all and, with an odd PAIRS,
    # handed the first-listed arm an extra position-1 slot.  The measured
    # position effect is +-18% per arm (~25% position-2 vs position-1), i.e.
    # the same size as the entire claim.  Rotation gives every arm every
    # position exactly PAIRS/n_arms times -- but only when PAIRS is a multiple
    # of n_arms, so that is enforced rather than hoped for.
    if ((n_arms > 1 && PAIRS % n_arms != 0)); then
        echo "ERROR: PAIRS=$PAIRS is not a multiple of the ${n_arms} arms." >&2
        echo "       Position balance would be nominal, not real: some arm" >&2
        echo "       gets more position-1 slots than another and the +-18%" >&2
        echo "       position effect lands straight on the headline ratio." >&2
        echo "       Use PAIRS=$(( (PAIRS / n_arms + 1) * n_arms )) (or any" >&2
        echo "       multiple of $n_arms), or set ALLOW_UNBALANCED_PAIRS=1 to" >&2
        echo "       run anyway -- which stamps POSITION_BALANCE=UNBALANCED" >&2
        echo "       into the campaign receipt and the analyzer will refuse to" >&2
        echo "       quote the ratios as a headline." >&2
        if ((${ALLOW_UNBALANCED_PAIRS:-0} == 0)); then
            exit 2
        fi
    fi
    claim_gpus
    mkdir -p "$ROOT"
    # F10 + F11: one receipt per invocation, so a second invocation into the
    # same ROOT is caught and so the audit can read the offered-rate provenance
    # and the balance verdict without reconstructing them from the log.
    local balance="BALANCED"
    if ((n_arms > 1 && PAIRS % n_arms != 0)); then balance="UNBALANCED"; fi
    if [[ -f "$ROOT/CAMPAIGN_INVOCATION.txt" ]] \
        && ! grep -qxF "spec: arms=$ARMS cells=$CELLS pairs=$PAIRS" \
            "$ROOT/CAMPAIGN_INVOCATION.txt"; then
        echo "ERROR: $ROOT was created by a DIFFERENT invocation:" >&2
        cat "$ROOT/CAMPAIGN_INVOCATION.txt" >&2
        echo "       Set RUN_TAG (or ROOT) for this run so the two campaigns" >&2
        echo "       cannot be fused into one pair tree by the analyzer." >&2
        exit 2
    fi
    {
        echo "spec: arms=$ARMS cells=$CELLS pairs=$PAIRS"
        echo "run_tag: $RUN_TAG"
        echo "prompt_source: $PROMPT_SOURCE qsl_pkl=${QSL_PKL:-none}"
        echo "prefix_caching: $PREFIX_CACHING (symmetric across all arms)"
        echo "position_balance: $balance (n_arms=$n_arms pairs=$PAIRS)"
        echo "open_loop_base_rate: ${OPEN_LOOP_BASE_RATE:-unset}"
        echo "open_loop_base_rate_source: ${OPEN_LOOP_BASE_RATE_SOURCE:-unset}"
        echo "arm_cooldown_s: ${ARM_COOLDOWN:-240}"
        echo "image: $IMAGE"
    } >"$ROOT/CAMPAIGN_INVOCATION.txt"

    for ((pair = 1; pair <= PAIRS; pair++)); do
        # Rotate by (pair-1) mod n_arms, then reverse every n_arms pairs so the
        # neighbour ordering is balanced too (arm X does not always follow Y).
        local -a order=()
        local block=$(( (pair - 1) / n_arms ))
        local shift_by=$(( (pair - 1) % n_arms ))
        for ((i = 0; i < n_arms; i++)); do
            order+=("${arm_list[(i + shift_by) % n_arms]}")
        done
        if ((block % 2 == 1)); then
            local -a reversed=()
            for ((i = ${#order[@]} - 1; i >= 0; i--)); do
                reversed+=("${order[i]}")
            done
            order=("${reversed[@]}")
        fi
        local position=0
        local pair_void=0
        for arm in "${order[@]}"; do
            position=$((position + 1))
            local tag
            printf -v tag '%02d' "$pair"
            local container="m15_${RUN_TAG}_p${tag}_${position}_${arm}"
            local output_dir="$ROOT/pair_${tag}/${position}_${arm}"
            mkdir -p "$output_dir"
            if ! reserve_container_name "$container" \
                || ! guard_output_dir "$output_dir"; then
                return 1
            fi
            printf '%s\n' "$(numerics_class "$arm")" \
                >"$output_dir/numerics_class.txt"
            # Uniform pre-arm cooldown: position-2 arms measured ~25%
            # slower than position-1 (thermal/host-state carryover);
            # equalize conditions across positions.
            sleep "${ARM_COOLDOWN:-240}"
            CURRENT_CONTAINER="$container"
            # F8.  A rejected startup flag on ANY arm used to kill the whole
            # driver through errexit, at (say) pair 3 of 5, leaving exactly the
            # unbalanced partial dataset this protocol exists to forbid.  An
            # arm that will not start now VOIDS ITS PAIR -- the analyzer drops
            # a pair carrying PAIR_VOID.txt, so what survives is a smaller but
            # still position-balanced set of pairs -- and the campaign goes on.
            if ! launch_server "$container" "$arm" "$output_dir" 0 0 \
                || ! wait_ready "$container" "$arm" "$output_dir"; then
                echo "ERROR: arm '$arm' failed to start in pair $tag" >&2
                {
                    echo "pair $tag voided: arm '$arm' (position $position)"
                    echo "did not reach readiness; see its server.log."
                } >"$ROOT/pair_${tag}/PAIR_VOID.txt"
                echo "ARM_START=FAIL" >"$output_dir/ARM_STATUS.txt"
                docker logs "$container" >"$output_dir/server.log" 2>&1 || true
                salvage_container
                pair_void=1
                continue
            fi
            echo "ARM_START=OK" >"$output_dir/ARM_STATUS.txt"
            # Fairness evidence BEFORE any measurement: image digest, the
            # container's real env, the resolved server config and the
            # patch-marker census.  A native arm that fails authenticity here
            # voids its pair rather than producing a number someone would later
            # have to retract.
            if ! capture_config_receipt "$container" "$arm" "$output_dir"; then
                echo "ERROR: authenticity receipt FAILED for '$arm'" >&2
                {
                    echo "pair $tag voided: arm '$arm' (position $position)"
                    echo "failed the untouched-image authenticity check."
                } >"$ROOT/pair_${tag}/PAIR_VOID.txt"
                echo "ARM_AUTHENTICITY=FAIL" >>"$output_dir/ARM_STATUS.txt"
                docker logs "$container" >"$output_dir/server.log" 2>&1 || true
                stop_server "$container" || true
                CURRENT_CONTAINER=""
                pair_void=1
                continue
            fi
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
                if ! run_cell "$container" "$arm" "$pair" "$cell" "$output_dir"; then
                    echo "ERROR: cell '$cell' failed on arm '$arm'; voiding pair" >&2
                    {
                        echo "pair $tag voided: cell '$cell' failed on arm"
                        echo "'$arm' (position $position)."
                    } >"$ROOT/pair_${tag}/PAIR_VOID.txt"
                    echo "CELL_${cell}=FAIL" >>"$output_dir/ARM_STATUS.txt"
                    pair_void=1
                fi
            done
            record_receipts "$container" "$barm" "$output_dir"
            docker logs "$container" >"$output_dir/server.log" 2>&1
            grep -iE "eplb|balancedness|rearrang" "$output_dir/server.log" \
                | tail -300 >"$output_dir/eplb_lines.txt" || true
            grep "M15_COVERAGE" "$output_dir/server.log" \
                | tail -64 >"$output_dir/coverage_tail.txt" || true
            stop_server "$container"
            CURRENT_CONTAINER=""
        done
        if ((pair_void)); then
            echo "WARNING: pair $pair is VOID; the analyzer will drop it." >&2
            echo "         Run one extra pair to keep n and the rotation" >&2
            echo "         intact (a void pair removes one full rotation slot)." >&2
        fi
    done

    # ACCURACY GATE before the diagnostic passes: it is the one pass that can
    # invalidate every number the pairs just produced, so it must not sit
    # behind a profiler that might fail first.
    if ((ACCURACY_PASS)); then
        local acc_root="$ROOT/accuracy"
        for arm in "${arm_list[@]}"; do
            if ! accuracy_pass "$arm" "$acc_root/${arm}"; then
                # NOT "continuing" in the v5.0 sense: the arm is still EXPECTED
                # by the gate below, so its absence is a FAIL rather than a
                # silently smaller comparison (F5, item 1).
                echo "ERROR: accuracy pass FAILED for arm $arm; the gate will" >&2
                echo "       report it as a missing arm, not skip it." >&2
                salvage_container
            fi
        done
        if assert_accuracy_gate "$acc_root" "${ACCURACY_CANDIDATE:-m15}" \
            "${arm_list[@]}"; then
            echo "ACCURACY GATE: PASS (see $acc_root/ACCURACY_GATE.txt)"
        else
            echo "########################################################" >&2
            echo "ACCURACY GATE: FAIL -- no performance claim may be made" >&2
            echo "from this campaign until parity is resolved." >&2
            echo "see $acc_root/ACCURACY_GATE.txt" >&2
            echo "########################################################" >&2
        fi
    fi

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
