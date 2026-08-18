#!/usr/bin/env python3
"""k0pf prefill movement gates and six-arm region A/B.

Additive executor harness derived from the frozen e004 prefill oracle. It keeps both MoRI stream
barriers and the capture-resident torch zero of ``part``. The k0pf plan, quant, gather, and combine
are individually sidecar-tested before the full candidate is timed.

FAIRNESS FIX (problem 11): the production arm previously ended with a TIMED base_out.copy_(res)
([T,H] bf16 D2D) that NO candidate arm had (candidates write moe_out in place). Production's real
EpCombine writes moe_out directly, so that copy was a pure harness artifact. Removed; production now
reads its combine output directly. All arms then have identical timed I/O boundaries. The removed
copy is separately micro-measured (copy_cost_us) so its magnitude is on the record.
Original FULL-PUSH LIFECYCLE region A/B.  ONE process, 8 ranks, region corpus, graph mode.

Forked from `tile/reference/e22_region_ab.py` (best_1075x host oracle).  The ONE VARIABLE is the
communication lifecycle at the two expert-region boundaries: frozen PULL with two exposed host
`shmem_barrier_on_stream` calls  vs  API source-PUSH with device fabric-completion tickets (§L58 MORI
flag-counter release/acquire).  Five arms, same-run, position-balanced, aligned MAX(rank):

  (a) production   = MORI EpDispatch(bf16) -> aiter fused_moe(fp8 blockscale) -> MORI EpCombine
  (b) frozen_pull  = quant -> barrier1 -> gather_pull -> n1g -> barrier2 -> combine_pull   (best_1075x, gblk=256/cblk=512)
  (c) disp_push    = PUSH dispatch (+ fabric ticket + acquire-copy) -> n1g -> barrier2 -> combine_pull
  (d) comb_push    = quant -> barrier1 -> gather_pull -> n1g -> PUSH combine (slots + ticket + acquire + reduce)
  (e) full_push    = PUSH dispatch -> n1g -> PUSH combine   (ZERO exposed host barriers)

The dispatch-only (c) and combine-only (d) sidecars are REQUIRED for attribution (plan §"The one
variable"): the primary hypothesis is the full lifecycle (e), but (c)/(d) isolate which boundary pays.

  >>> PHASE-A STATUS <<<  The three push kernels (k0_tile_dispatch_push / k0_tile_combine_push /
  k0_tile_expected_mask) are authored in Phase B on the gfx950 node.  This harness loads them in a
  try/except: if the `k0_tile_push` JIT module is absent, arms (c)(d)(e) are marked STUBBED and only
  (a)(b) run — so Phase-A can smoke-test production vs frozen-pull, and Phase B lights up the push
  arms by dropping the compiled kernels in place.  Every push-arm launch site is marked  # PUSH:.

RULES §4/§5/§12:  ALL candidate-unique work (route all-gather, the 9-kernel plan, origin quant, the
NEW expected-mask prologue / epoch bump / push / release / acquire / local reduce) stays INSIDE the
timed graph.  3 capture warmups, correctness gate on every arm (frozen tolerance, never widened), 20
untimed replays, 100 timed position-balanced rotations, aligned MAX(rank), p50/p95 + every ratio.

Launch: torchrun --standalone --nnodes=1 --nproc_per_node=8 e001_push_ab.py
"""
import hashlib, json, os, re, sys, shutil, traceback
os.environ.setdefault("HSA_XNACK", "1")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_K0_ROOT = os.path.abspath(
    os.environ.get("K0_BASE", os.path.join(_THIS_DIR, "..", ".."))
)
_MOK_BENCH_DIR = os.path.join(_K0_ROOT, "benchmarks", "mok_synthetic_prefill")
sys.path.insert(0, os.environ.get("K0_PYBIND_DIR", os.path.join(_MOK_BENCH_DIR, "build", "python")))
sys.path.insert(0, _MOK_BENCH_DIR)
import numpy as np
import torch
import torch.distributed as dist
import mori
import mori.shmem as ms
import tile_plan, k0pf_frozen_plan, k0pf_plan, k0_n1g, k0_mps_host_abi
from correctness import get_mok_error_stats, mok_error_passes
from synthetic_inputs import MoKSyntheticConfig, generate_mok_synthetic_inputs
try:
    import k0pf2_plan
    _HAS_PF2_PLAN = True
    _PF2_PLAN_ERR = None
except Exception as _e_pf2_plan:
    _HAS_PF2_PLAN = False
    _PF2_PLAN_ERR = repr(_e_pf2_plan)
try:
    import k0_n2                                   # exp_47 two-phase (split-N full-K W2, write-once) sibling
    _HAS_N2 = True; _N2_ERR = None
except Exception as _e_n2:
    _HAS_N2 = False; _N2_ERR = repr(_e_n2)
try:
    import k0_n1g_c32                              # c32 tail-aware sibling (BM32-M16 skip); drop-in, oracle untouched
    _HAS_C32 = True; _C32_ERR = None
except Exception as _e_c32:
    _HAS_C32 = False; _C32_ERR = repr(_e_c32)
try:
    import k0_n1g_ah                               # AH W13 address-hoist sibling; drop-in, oracle untouched
    _HAS_AH = True; _AH_ERR = None
except Exception as _e_ah:
    _HAS_AH = False; _AH_ERR = repr(_e_ah)
try:
    import k0_n1g_ecs                              # ECS full-run co-scheduling sibling (lower-bound artifact; has an AGPR drain); drop-in
    _HAS_ECS = True; _ECS_ERR = None
except Exception as _e_ecs:
    _HAS_ECS = False; _ECS_ERR = repr(_e_ecs)
try:
    import k0_n1g_ecs_d2                           # ECS-D2 DECISIVE sibling (pair-leader D=2, per-block e reload, no drain); drop-in
    _HAS_ECS_D2 = True; _ECS_D2_ERR = None
except Exception as _e_ecs_d2:
    _HAS_ECS_D2 = False; _ECS_D2_ERR = repr(_e_ecs_d2)
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from synthetic_routes import FAMILIES as SYNTH_ROUTE_FAMILIES, generate_synthetic_route
# ON-TARGET COMPUTE-SWAP BRIDGE (arm f): replicate EXACTLY what aiter.fused_moe does internally
# before the GEMM — moe_sorting (global E, expert_mask) then per_1x128 quant with transpose_scale=True
# (group-major scale, mandatory for n1g) — then call n1g instead of fmoe_fp8_blockscale_g1u1.
# get_hip_quant is what fused_moe.py imports as get_quant (see pinned source line 16).
_hipquant = aiter.get_hip_quant(QuantType.per_1x128)

WORLD = int(os.environ.get("WORLD_SIZE", "1")); WR = int(os.environ.get("RANK", "0")); LR = int(os.environ.get("LOCAL_RANK", "0"))
TOPK = 8; E = 32; E_GLOBAL = 256; K = H = 7168; INTER = 2048; NG = 56
# Regime-shaped capacities. Defaults are the frozen DECODE values -- every decode result stays
# byte-reproducible with no env set. Prefill overrides them via K0_* (see tile/ledger.md).
#   T          batch (decode 64, prefill 4096)
#   T_LOC_MAX  rows this rank RECEIVES = WORLD*T  (decode 512, prefill 32768)
#              Arrival-side, not send-side: gath[T_LOC_MAX,2] holds (src_rank, src_idx) per LOCAL row,
#              fn_gather launches (T_LOC_MAX,) blocks writing local a_dst, part[:T_LOC_MAX] is n1g's
#              output region. Bound is WORLD*T because MoRI DEDUPS per destination (L86,
#              intranode.hpp:104-116) -- the same structural fact that makes MAXTOK >= T safe.
#              NOTE: T*TOPK gives the same number here ONLY because WORLD == TOPK == 8. That form
#              counts routed rows BEFORE dedup and is wrong on any other topology. Use WORLD*T.
#   PADMAX     sorted-token capacity, including up to 31 padding rows per local expert
#   MROWS      expert-input row capacity, post-TOPK-expansion = WORLD*T*TOPK (decode 4096)
#   MAXTOK     MoRI max_num_inp_token_per_rank -- bounds INPUT tokens/rank (= T), not routed rows
T = int(os.environ.get("K0_T", "4096"))
T_LOC_MAX = int(os.environ.get("K0_T_LOC_MAX", "40960"))
if T_LOC_MAX < WORLD * T:
    raise ValueError(
        f"K0_T_LOC_MAX={T_LOC_MAX} is unsafe for direct push; require >= WORLD*T={WORLD*T}"
    )
# M18 static hot-expert replication (mps_mega arm only, mok_synthetic mode
# only).  Value: an integer K (top-K experts of the K0_MOK_ROUTE_HIST file)
# or an explicit comma-separated global-expert-id list.  The kernel arm is
# ablations-m18 (K0P6_M15_REPLICATE); replicated experts dispatch source-
# locally and every rank hosts their weights in slots E..E+nrep-1.
K0_MOK_REP_EXPERTS = os.environ.get("K0_MOK_REP_EXPERTS", "").strip()
# M19: per-chunk adaptive decision threshold (own routed-slot count).
# 0 = plain M18 (always replicate).  >0 requires the m19 kernel arm
# (K0P6_M15_ADAPTIVE): 65-word descriptor + symmetric rep_dec words.
K0_MOK_REP_THRESHOLD = int(os.environ.get("K0_MOK_REP_THRESHOLD", "0"))
# M20 slot-pool arm: replica weights in a [3][B] pool, prefetch duty
# streamed in-kernel from symmetric export staging, decisions
# precomputed (the fixed route makes the bitmap constant here).
K0_MOK_M20 = os.environ.get("K0_MOK_M20", "").strip() == "1"
if K0_MOK_M20 and not (K0_MOK_REP_EXPERTS and K0_MOK_REP_THRESHOLD):
    raise ValueError("K0_MOK_M20 requires K0_MOK_REP_EXPERTS and K0_MOK_REP_THRESHOLD")
if K0_MOK_REP_THRESHOLD and not K0_MOK_REP_EXPERTS:
    raise ValueError("K0_MOK_REP_THRESHOLD requires K0_MOK_REP_EXPERTS")
_M18_NREP = 0
if K0_MOK_REP_EXPERTS:
    _m18_csv = "," in K0_MOK_REP_EXPERTS
    _M18_NREP = (
        len([x for x in K0_MOK_REP_EXPERTS.split(",") if x.strip()])
        if _m18_csv
        else int(K0_MOK_REP_EXPERTS)
    )
    if not 1 <= _M18_NREP <= 32:
        raise ValueError(f"K0_MOK_REP_EXPERTS must name 1..32 experts, got {_M18_NREP}")

PADMAX_SAFE = WORLD * T * TOPK + (32 - 1) * E
if _M18_NREP:
    # replica slots pad like owned experts: up to 31 rows each, kept 32-aligned
    PADMAX_SAFE += ((31 * _M18_NREP + 31) // 32) * 32
PADMAX = int(os.environ.get("K0_PADMAX", str(PADMAX_SAFE)))
if PADMAX < PADMAX_SAFE or PADMAX % 32:
    raise ValueError(
        f"K0_PADMAX={PADMAX} is unsafe; require a 32-row multiple >= {PADMAX_SAFE}"
    )
MROWS = T_LOC_MAX                      # expert-input row capacity follows T_LOC_MAX at prefill
MAXTOK = int(os.environ.get("K0_MAXTOK", "4096"))        # candidate arms: MAXTOK = T (banked win, L86)
MAXTOK_PROD = int(os.environ.get("K0_MAXTOK_PROD", "32768"))  # production arm keeps the SHIPPED value
NTIMED = int(os.environ.get("K0_NTIMED", "100")); NUNTIMED = 20
K0_INPUT_MODE = os.environ.get("K0_INPUT_MODE", "capture").strip()
K0_BENCHMARK_PROTOCOL = os.environ.get("K0_BENCHMARK_PROTOCOL", "graph").strip()
if K0_INPUT_MODE not in ("capture", "mok_synthetic"):
    raise ValueError("K0_INPUT_MODE must be capture or mok_synthetic")
if K0_MOK_REP_EXPERTS and K0_INPUT_MODE != "mok_synthetic":
    raise ValueError("K0_MOK_REP_EXPERTS requires K0_INPUT_MODE=mok_synthetic")
# M18 state, populated in the mok_synthetic block when replication is on.
_m18_rep_ids = None
_m18_rep_buf = None
_m18_w13 = _m18_fc1 = _m18_w2c = _m18_fc2 = None
if K0_BENCHMARK_PROTOCOL not in ("graph", "mok_eager"):
    raise ValueError("K0_BENCHMARK_PROTOCOL must be graph or mok_eager")
if K0_BENCHMARK_PROTOCOL == "mok_eager" and K0_INPUT_MODE != "mok_synthetic":
    raise ValueError("mok_eager protocol requires K0_INPUT_MODE=mok_synthetic")
MOK_WARMUP_ITERS = int(os.environ.get("K0_MOK_WARMUP_ITERS", "500"))
MOK_TIMED_ITERS = int(os.environ.get("K0_MOK_TIMED_ITERS", "100"))
MOK_ABSOLUTE_TOLERANCE = float(
    os.environ.get("K0_MOK_ABSOLUTE_TOLERANCE", "1.0")
)
MOK_RELATIVE_TOLERANCE = float(
    os.environ.get("K0_MOK_RELATIVE_TOLERANCE", "0.1")
)
if MOK_WARMUP_ITERS < 0 or MOK_TIMED_ITERS <= 0:
    raise ValueError("MoK warmup must be nonnegative and timed iterations must be positive")
if MOK_ABSOLUTE_TOLERANCE < 0 or MOK_RELATIVE_TOLERANCE < 0:
    raise ValueError("MoK correctness tolerances must be nonnegative")
# exp_32 gate hardening. `cand_out` is cleared once per gate episode but NEVER
# between epochs, and the MoK corpus feeds identical input and identical routing
# every iteration, so a row a candidate fails to write returns the previous
# epoch's bit-identical correct answer -- invisible to [MOK GATE], to
# combine_bit_exact, and to 600 soak epochs. Poison with a value that cannot
# survive a correct run; correctness.py's `nonfinite == 0` gate then catches it.
MOK_POISON_OUT = os.environ.get("K0_MOK_POISON_OUT", "1").strip() not in ("", "0")
# bf16 quiet NaN, payload 0x55. NOT 0x7FC0 (what a computed 0/0 rounds to) and
# NOT 0x7F80 (inf), so a survivor is distinguishable from a kernel-produced NaN.
MOK_POISON_I16 = 0x7FD5
GBLK = int(os.environ.get("K0_GBLK", "256"))   # frozen gather block (best_1075x)
CBLK = int(os.environ.get("K0_CBLK", "512"))   # frozen combine block (best_1075x)
NSLOT = 2                                        # double-buffered epoch slots
M_TRIM = T * TOPK * WORLD
OUT = os.environ.get(
    "MORI_OUT_DIR",
    os.environ.get(
        "K0_OUT_DIR", os.path.join(_MOK_BENCH_DIR, "results", "manual")
    ),
)
HIP_SRC = os.environ.get(
    "K0_HIP_SRC", os.path.join(_K0_ROOT, "tile", "kernels", "k0_region_kernels_e22.hip")
)
PUSH_SRC = os.environ.get(
    "K0_PUSH_SRC", os.path.join(_K0_ROOT, "tile", "kernels", "k0_tile_push_kernels.hip")
)  # PHASE B
REGION_CORPUS = os.environ.get("K0_REGION_CORPUS",
    "/data/subvadla-traces/production-k0-region-capture/20260714T215732Z_k0_region_capture")
KERNELS_DIR = os.environ.get(
    "K0_MORI_KERNELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(mori.__file__)), "_jit-sources", "src", "ops", "kernels"),
)
KNAME = "k0_region_kernels_e22"
PF_HSACO_DIR = os.environ.get(
    "K0_PF_HSACO_DIR", os.path.join(_K0_ROOT, "prefill_opt", "hsaco")
)
PF2_HSACO = os.environ.get(
    "K0_PF2_HSACO", "/data/ecs-d2-ab/build_k0pf2_20260727/k0pf2_push.hsaco"
)
PF2_PUSH_GRID = int(os.environ.get("K0_PF2_PUSH_GRID", str(max(1, min(2048, (T + 3) // 4)))))
PF2_TRANSPOSE_GRID = int(
    os.environ.get("K0_PF2_TRANSPOSE_GRID", str(max(1, min(256, (T_LOC_MAX + 3) // 4))))
)
PNAME = "k0_tile_push_kernels"     # PHASE B combined JIT unit (dispatch_push + combine_push + expected_mask)
# k0d is intentionally a B64-only decode package.  Keep its source lookup relative to this
# harness so a copied node checkout never needs a second, divergent path configuration.
K0D_NAMES = (
    "k0d_fused", "k0d_barrier", "k0d_gather_zero", "k0d_combine",
    "k0d_push", "k0d_dsort", "k0d_mega", "k0d_phase_marker",
)
K0D_KERNEL_DIR = os.environ.get(
    "K0D_SRC_DIR", os.environ.get(
        "K0D_KERNEL_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels"))
    )
)
K0D_REQUESTED = bool({"k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f", "k0d_pd", "k0d_mega"} & {
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",") if nm
})
# pf4h is the hierarchical-histogram destination sort. It reuses the entire PF3 lane — module
# loading, disjoint symmetric protocol state, primitive gates, epoch poison — and differs from
# pf3_pd only in which destination-sort kernel the composition launches.
PF3_ARM_NAMES = ("pf3_pd", "pf3_mega", "pf4h")
PF3_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF3_ARM_NAMES
)
PF3_REQUESTED = bool(PF3_REQUESTED_ARMS)
PF4H_REQUESTED = "pf4h" in PF3_REQUESTED_ARMS
PF5_ARM_NAMES = ("pf5_ll",)
PF5_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF5_ARM_NAMES
)
PF5_REQUESTED = bool(PF5_REQUESTED_ARMS)
PF6_ARM_NAMES = ("pf6_mega",)
PF6_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF6_ARM_NAMES
)
# exp_56 Option C: chunk-counted release arm (additive sibling to pf6_mega).
PF6C_ARM_NAMES = ("pf6c_mega",)
PF6C_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF6C_ARM_NAMES
)
PF6C_REQUESTED = bool(PF6C_REQUESTED_ARMS)
# exp_59: large-M G-stacked expert tiles arm (additive sibling; N2 kBlockM 32 -> 32*G).
PF6GM_ARM_NAMES = ("pf6gm_mega",)
PF6GM_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF6GM_ARM_NAMES
)
PF6GM_REQUESTED = bool(PF6GM_REQUESTED_ARMS)
PF6MPS_ARM_NAMES = ("mps_mega",)
PF6MPS_REQUESTED_ARMS = tuple(
    nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",")
    if nm in PF6MPS_ARM_NAMES
)
PF6MPS_REQUESTED = bool(PF6MPS_REQUESTED_ARMS)
# exp_59: G-stack width. MUST match K0P6GM_G in k0pf6gm_mega.hip AND N2GM_G in the _gm bodies.
# Default 2 (M=64): the FUSED megakernel build gate at G=3 (M=96) spills 7 VGPR / 32B scratch
# because the phase UNION (dispatch+verify+N2+combine) shares one static allocation, tipping
# the 256-AGPR wall — even though N2-only G=3 is clean. G=2 is the largest clean FUSED rung.
# K0_PF6GM_G is NOT forwarded by the frozen run_campaign.sh, so this source default governs the
# container build; override locally only for direct (non-campaign) launches.
PF6GM_G = int(os.environ.get("K0_PF6GM_G", "2"))
PF6MPS_G = 3
if PF6MPS_REQUESTED and PF6GM_G != PF6MPS_G:
    raise RuntimeError(
        f"mps_mega requires its paired pf6gm_mega reference at G={PF6MPS_G}; "
        f"got K0_PF6GM_G={PF6GM_G}"
    )
# The shared PF6 buffer/N2 machinery must be built whenever ANY pf6 arm is requested.
PF6_REQUESTED = (
    bool(PF6_REQUESTED_ARMS) or PF6C_REQUESTED or PF6GM_REQUESTED
    or PF6MPS_REQUESTED
)
PF3_NAMES = ("k0pf3_qpush", "k0pf3_dsort", "k0pf3_mega") + (
    ("k0pf4_dsort",) if PF4H_REQUESTED else ()
)
PF3_DIAG_NAMES = ("k0pf3_qpush_ts", "k0pf3_dsort_ts")  # loaded only when K0_PF3_TS=1
# k0pf4_dsort is the first kernel here built on the hkp header tier. The JIT compiles out of
# KERNELS_DIR, so the headers are installed there as flat siblings — the same mechanism
# k0_tile_push_kernels already uses for its #included sub-sources. mori's JIT cache key hashes
# every .hpp under that tree, so editing a header still invalidates the compiled object.
HKP_HEADER_DIR = os.environ.get(
    "K0_HKP_DIR",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "solution", "hip", "hkp")
    ),
)
HKP_HEADERS = ("hkp_topology.hpp", "hkp_sync.hpp", "hkp_sort.hpp", "hkp_quant.hpp")
PF3_KERNEL_DIR = os.environ.get(
    "K0PF3_SRC_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels"))
)
PF3_QPUSH_GRID = int(os.environ.get("K0_PF3_QPUSH_GRID", "1024"))
PF3_DSORT_GRID = int(os.environ.get("K0_PF3_DSORT_GRID", "64"))
PF3_MEGA_GRID = int(os.environ.get("K0_PF3_MEGA_GRID", "512"))
PF4_DSORT_GRID = int(os.environ.get("K0_PF4_DSORT_GRID", str(PF3_DSORT_GRID)))
PF5_QPUSH_GRID = int(os.environ.get("K0_PF5_QPUSH_GRID", str(PF3_QPUSH_GRID)))
PF5_DSORT_GRID = int(os.environ.get("K0_PF5_DSORT_GRID", str(PF3_DSORT_GRID)))
PF5_NAMES = ("k0pf5_qpush_ll", "k0pf5_dsort_ll")
PF5_HEADERS = ("k0pf5_ll128.hpp",)
PF5_ROW_STRIDE = 8064
PF6_NAME = "k0pf6_mega"
PF6C_NAME = "k0pf6c_mega"   # exp_56 Option C chunk-counted-release megakernel
PF6GM_NAME = "k0pf6gm_mega"  # exp_59 large-M G-stacked expert-tile megakernel
PF6MPS_NAME = "k0pf6gm_mps_mega"
PF6MPS_SNAPSHOT_NAME = "k0pf6_mori_heap_snapshot"
PF6_MEGA_GRID = 256
PF6_THREADS = 256
PF6_N2_SOURCE_DIR = os.environ.get(
    "K0_PF6_N2_SRC_DIR",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "solution", "hip")
    ),
)
PF6_HIPKITTENS_ROOT = os.environ.get(
    "K0_PF6_HIPKITTENS_ROOT",
    os.path.abspath(os.path.join(_K0_ROOT, "..", "..", "HipKittens")),
)
PF6MPS_DHK_ROOT = os.path.abspath(
    os.environ.get(
        "K0_DHK_ROOT",
        os.path.join(os.path.expanduser("~"), "Distributed-HipKittens"),
    )
)
PF6MPS_SOURCE_DIR = os.path.join(
    PF6MPS_DHK_ROOT, "distributed-kernels", "fused_moe"
)
PF6_N2_FILES = (
    "n2_phase1.cpp",
    "n2_phase2.cpp",
    "n2_fused_moe.hpp",
    "n2_device_common.cuh",
    "aiter_gfx950_fp8_layout.hpp",
) + (
    ("n2_phase1_gm.cpp",)
    if PF6GM_REQUESTED or PF6MPS_REQUESTED else ()
) + (
    ("n2_phase2_gm.cpp",) if PF6GM_REQUESTED else ()
) + (
    ("n2_phase2_gm_mps.cpp",) if PF6MPS_REQUESTED else ()
) + (
    # exp_26: vendored phase-1 body for mps_mega. The donor n2_phase1_gm.cpp
    # above STAYS installed -- pf6gm_mega still includes it.
    ("n2_phase1_gm_mps.cpp",) if PF6MPS_REQUESTED else ()
)

def _assert_gstack_source_contract(label, hip_path, phase_paths, expected_g):
    with open(hip_path, "r", encoding="utf-8") as source_file:
        hip_text = source_file.read()
    required_hip = (
        "#define N2GM_G K0P6GM_G",
        "#ifndef K0P6GM_G",
    )
    missing = [token for token in required_hip if token not in hip_text]
    if missing:
        raise RuntimeError(
            f"{label} source does not bind N2GM_G to K0P6GM_G={expected_g}: "
            f"missing {missing} in {hip_path}"
        )
    pinned = re.search(r"#if\s+K0P6GM_G\s*!=\s*(\d+)", hip_text)
    if pinned is not None and int(pinned.group(1)) != expected_g:
        raise RuntimeError(
            f"{label} source pins K0P6GM_G={pinned.group(1)} but "
            f"PF6GM_G={expected_g}"
        )
    for phase_path in phase_paths:
        with open(phase_path, "r", encoding="utf-8") as source_file:
            phase_text = source_file.read()
        if "constexpr int kGM = N2GM_G" not in phase_text:
            raise RuntimeError(
                f"{label} phase body does not consume N2GM_G: {phase_path}"
            )
    return {
        "label": label,
        "PF6GM_G": int(expected_g),
        "K0P6GM_G": int(expected_g),
        "N2GM_G": int(expected_g),
        "hip_source": hip_path,
        "phase_sources": list(phase_paths),
    }

PF6_GSTACK_SOURCE_CONTRACTS = []
if PF6GM_REQUESTED:
    PF6_GSTACK_SOURCE_CONTRACTS.append(
        _assert_gstack_source_contract(
            "pf6gm_mega",
            os.path.join(PF3_KERNEL_DIR, PF6GM_NAME + ".hip"),
            (
                os.path.join(PF6_N2_SOURCE_DIR, "n2_phase1_gm.cpp"),
                os.path.join(PF6_N2_SOURCE_DIR, "n2_phase2_gm.cpp"),
            ),
            PF6GM_G,
        )
    )
if PF6MPS_REQUESTED:
    PF6_GSTACK_SOURCE_CONTRACTS.append(
        _assert_gstack_source_contract(
            "mps_mega",
            os.path.join(PF6MPS_SOURCE_DIR, "k0pf6gm_device_tile_mps.hip"),
            (
                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase1_gm_mps.cpp"),
                os.path.join(PF6MPS_SOURCE_DIR, "n2_phase2_gm_mps.cpp"),
            ),
            PF6MPS_G,
        )
    )

def _parse_mps_config(text):
    fields = {}
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid K0_MPS_CFG item {item!r}")
        key, value = (part.strip() for part in item.split("=", 1))
        if key in fields:
            raise ValueError(f"duplicate K0_MPS_CFG field {key!r}")
        fields[key] = int(value, 0)
    required = {"C", "g", "mode", "flush_rows"}
    optional = {"pull_fallback", "timestamps"}
    if set(fields) - required - optional or required - set(fields):
        raise ValueError(
            "K0_MPS_CFG requires C,g,mode,flush_rows and optionally "
            f"pull_fallback,timestamps; got {sorted(fields)}"
        )
    config = {
        "C": fields["C"],
        "g": fields["g"],
        "mode": fields["mode"],
        "flush_rows": fields["flush_rows"],
        "pull_fallback": fields.get("pull_fallback", 0),
        "timestamps": fields.get("timestamps", 0),
    }
    if config["pull_fallback"] not in (0, 1) or config["timestamps"] not in (0, 1):
        raise ValueError("K0_MPS_CFG flags must be 0 or 1")
    config["word"] = int(
        k0_mps_host_abi.encode_config(
            config["C"], config["g"], config["mode"], config["flush_rows"],
            bool(config["pull_fallback"]), bool(config["timestamps"]),
        )
    )
    return config

PF6MPS_CONFIG = (
    _parse_mps_config(os.environ.get("K0_MPS_CFG", "").strip())
    if PF6MPS_REQUESTED else None
)
# ---- exp_02 (aug12, mode 16 deferred combine) generation-parity support ----
# Fully inert unless the packed mode is 16: TBO16_G collapses to 1 and every
# gate/poison path keeps its original shape and indexing.
MPS_MODE16 = bool(PF6MPS_REQUESTED and PF6MPS_CONFIG["mode"] == 16)
TBO16_G = 2 if MPS_MODE16 else 1
_tbo16_launches = 0   # mps_mega launches executed this process
PF4_MAXE = 64      # K0P4_MAXE in k0pf4_dsort.hip
PF4_MAXCTA = 256   # K0P4_MAXCTA in k0pf4_dsort.hip
K0_ROUTE_SWAP_CORPUS = os.environ.get(
    "K0_ROUTE_SWAP_CORPUS",
    "/data/ecs-d2-production-captures/decode/tp1_dp8_ep8/B64/skewed_20260723T214320Z",
)
K0_SYNTH_ROUTE = os.environ.get("K0_SYNTH_ROUTE", "").strip()
K0_SYNTH_SEED = int(os.environ.get("K0_SYNTH_SEED", "0"))
K0_SYNTH_WEIGHT_MODE = os.environ.get("K0_SYNTH_WEIGHT_MODE", "captured").strip()
if K0_SYNTH_ROUTE and K0_SYNTH_ROUTE not in SYNTH_ROUTE_FAMILIES:
    raise ValueError(
        f"K0_SYNTH_ROUTE={K0_SYNTH_ROUTE!r} is invalid; choose one of {SYNTH_ROUTE_FAMILIES}"
    )
if K0_SYNTH_WEIGHT_MODE not in ("captured", "generated"):
    raise ValueError("K0_SYNTH_WEIGHT_MODE must be captured or generated")
# ---- ADDITIVE decode diagnostics (never part of any A/B arm or timed comparison) ----
# K0_DB_MICRO=1: doorbell/rendezvous mechanism microbenchmark (k0d_db_micro.hip).
# K0_MEGA_TS=1: internally timestamped k0d_mega twin replayed as a diagnostic graph.
# K0_SKIP_TIMED_AB=1: skip ONLY the 2xNTIMED timed A/B loops (all correctness/protocol
# gates, stress, poison, and route-swap still run before diagnostics).
_K0_DB_MICRO = bool(int(os.environ.get("K0_DB_MICRO", "0")))
_K0_MEGA_TS = bool(int(os.environ.get("K0_MEGA_TS", "0")))
_K0_SKIP_TIMED = bool(int(os.environ.get("K0_SKIP_TIMED_AB", "0")))
_K0_PF3_TS = bool(int(os.environ.get("K0_PF3_TS", "0")))
_K0_PF3_TS_REPS = int(os.environ.get("K0_PF3_TS_REPS", "30"))
# K0_PF4_TS=1: run the SAME diagnostic block but with the k0pf4_dsort_ts twin (schema 4,
# 384 slots) in place of k0pf3_dsort_ts. Thesis §5.2 / §9 #22 / §13.6 step 2 — the deployed
# dsort measures 1.589 ms in the live trace against 276.82 us standalone, and this is the
# measurement that decides whether that gap is doorbell spin or sort work. It implies
# K0_PF3_TS because the qpush twin supplies the same-rank "pushes finished" reference stamp.
_K0_PF4_TS = bool(int(os.environ.get("K0_PF4_TS", "0")))
if _K0_PF4_TS:
    _K0_PF3_TS = True
    # The twin needs st["hcnt"], which is only allocated when a pf3/pf4h arm is requested
    # (see the `if PF3_REQUESTED:` state fill-in). Fail here rather than KeyError inside a
    # launch, and fail rather than silently measure nothing.
    if not [
        nm for nm in os.environ.get("K0_ARMS", "").split(",")
        if nm.startswith("pf3") or nm == "pf4h" or nm == "pf3_pd"
    ]:
        raise RuntimeError(
            "K0_PF4_TS=1 requires a pf3/pf4h arm in K0_ARMS (it reuses that arm's "
            "hcnt/pull_stage/scratch state); e.g. K0_ARMS=production,pf4h"
        )
_PF_TS_SLOTS = 384 if _K0_PF4_TS else 256
_PF_TS_SCHEMA = 4 if _K0_PF4_TS else 2
# Diagnostic k0d modules are regime-independent: the doorbell/fabric microbenchmark measures
# the same hardware at prefill, and k0d_phase_marker provides the region phase stamps and the
# s_memrealtime calibration timer for every diagnostic block. Load them standalone when a
# prefill diagnostic is requested without any k0d decode arm.
_K0D_STANDALONE = (not K0D_REQUESTED) and (
    _K0_DB_MICRO
    or _K0_PF3_TS
    or bool(int(os.environ.get("K0_DECODE_PHASE_PROFILE", "0")))
)
K0D_DIAG_NAMES = tuple(
    nm for nm, on in (("k0d_db_micro", _K0_DB_MICRO), ("k0d_mega_ts", _K0_MEGA_TS)) if on
)
R = {
    "rank": WR,
    "pf6_gstack_source_contracts": PF6_GSTACK_SOURCE_CONTRACTS,
    "mps_config": PF6MPS_CONFIG,
}
os.makedirs(OUT, exist_ok=True)

torch.cuda.set_device(LR)
R["environment"] = {
    "host": os.environ.get("K0_HOSTNAME", os.uname().nodename),
    "container_image": os.environ.get("K0_CONTAINER_IMAGE"),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "device_name": torch.cuda.get_device_name(LR),
    "device_count": torch.cuda.device_count(),
    "mori_path": os.path.abspath(mori.__file__),
    "aiter_path": os.path.abspath(aiter.__file__),
}
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29655")
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=WR, world_size=WORLD)
from torch.distributed.distributed_c10d import _register_process_group, _get_default_group
try: dist.distributed_c10d._resolve_process_group("default")
except Exception: _register_process_group("default", _get_default_group())
ms.shmem_torch_process_group_init("default")
rank, world = ms.shmem_mype(), ms.shmem_npes()
cu = torch.cuda.get_device_properties(LR).multi_processor_count
BLK = int(os.environ.get("K0_BLK", "128")); WARP = int(os.environ.get("K0_WARP", "16"))
def hbarrier():
    torch.cuda.synchronize(); dist.barrier(); ms.shmem_barrier_all()

# k0d_fused owns a decode-specialized LDS plan and the arm-2 combine protocol has a fixed
# 16-block denominator.  Do not silently run it in a shape regime it was never designed for.
K0D_B64_PREFLIGHT = dict(
    world=world, T=T, T_LOC_MAX=T_LOC_MAX, PADMAX=PADMAX, MROWS=MROWS,
    MAXTOK=MAXTOK, MAXTOK_PROD=MAXTOK_PROD, NCHUNK=(WORLD * T * TOPK + 255) // 256,
)
K0D_B64_EXPECTED = dict(
    world=8, T=64, T_LOC_MAX=512, PADMAX=5088, MROWS=512,
    MAXTOK=512, MAXTOK_PROD=512, NCHUNK=16,
)
K0D_B64_OK = all(K0D_B64_PREFLIGHT[k] == v for k, v in K0D_B64_EXPECTED.items())
R["k0d"] = dict(
    requested=K0D_REQUESTED, b64_only=True, shape_preflight=K0D_B64_PREFLIGHT,
    shape_expected=K0D_B64_EXPECTED, shape_ok=K0D_B64_OK,
    modules_loaded=False, route_swap_corpus=K0_ROUTE_SWAP_CORPUS,
)
if K0D_REQUESTED and not K0D_B64_OK:
    raise RuntimeError(
        "k0d decode arms are B64-only; shape preflight failed: "
        f"got={K0D_B64_PREFLIGHT} expected={K0D_B64_EXPECTED}"
    )
if (PF3_REQUESTED or PF5_REQUESTED or PF6_REQUESTED) and T <= 512:
    raise RuntimeError(f"pf3/pf4h/pf5/pf6 arms are prefill-only; got T={T}")

# ===================== JIT: frozen kernels (always) + push kernels (Phase B) =====================
if rank == 0:
    shutil.copyfile(HIP_SRC, os.path.join(KERNELS_DIR, KNAME + ".hip"))
dist.barrier()
from mori.ops._jit_loader import ensure_compiled, load_hip_module
from mori.ops import _jit_loader as _k0d_jit_loader
ensure_compiled(KNAME)
mod = load_hip_module(KNAME, init_shmem=True)
fn_ag = mod.get_function("k0_region_ag_route")
fn_gather = mod.get_function("k0_region_gather_rows_fp8")
fn_combine = mod.get_function("k0_region_combine")
fn_combine_bug = mod.get_function("k0_region_combine_push_store")
fn_quant = mod.get_function("k0_region_quant_to_shmem")

# ---- k0d decode package.  It is opt-in through K0_ARMS and every module needs its own
# shmem-module initialization: all four use ShmemPtrP2p and/or ShmemGetMemBlock. ----
K0D_OK = False
fn_k0d_fused = fn_k0d_barrier = fn_k0d_gather_zero = fn_k0d_combine = None
fn_k0d_push = fn_k0d_dsort = fn_k0d_mega = None
fn_k0d_phase_marker = fn_k0d_phase_timer_calibration = None
fn_k0d_db_micro = fn_k0d_mega_ts = None
_mega_ts_buf = None
if K0D_REQUESTED:
    _K0D_JIT_NAMES = list(K0D_NAMES) + [
        nm for nm in K0D_DIAG_NAMES if nm not in K0D_NAMES
    ]
elif _K0D_STANDALONE:
    _K0D_JIT_NAMES = []
    if _K0_DB_MICRO:
        _K0D_JIT_NAMES.append("k0d_db_micro")
    _K0D_JIT_NAMES.append("k0d_phase_marker")
else:
    _K0D_JIT_NAMES = []
if _K0D_JIT_NAMES:
    _k0d_copy_err = [None]
    if rank == 0:
        try:
            for _k0d_name in _K0D_JIT_NAMES:
                _k0d_src = os.path.join(K0D_KERNEL_DIR, _k0d_name + ".hip")
                if not os.path.isfile(_k0d_src):
                    raise FileNotFoundError(_k0d_src)
                shutil.copyfile(_k0d_src, os.path.join(KERNELS_DIR, _k0d_name + ".hip"))
        except Exception as _e_k0d_copy:
            _k0d_copy_err[0] = "".join(
                traceback.format_exception_only(type(_e_k0d_copy), _e_k0d_copy)
            ).strip()
    dist.broadcast_object_list(_k0d_copy_err, src=0)
    if _k0d_copy_err[0] is not None:
        raise RuntimeError(f"k0d source install failed: {_k0d_copy_err[0]}")
    dist.barrier()
    try:
        for _k0d_name in _K0D_JIT_NAMES:
            ensure_compiled(_k0d_name)
        _k0d_mods = {
            _k0d_name: load_hip_module(
                _k0d_name, init_shmem=(_k0d_name != "k0d_phase_marker")
            )
            for _k0d_name in _K0D_JIT_NAMES
        }
        if "k0d_fused" in _k0d_mods:
            fn_k0d_fused = _k0d_mods["k0d_fused"].get_function("k0d_fused")
            fn_k0d_barrier = _k0d_mods["k0d_barrier"].get_function("k0d_barrier")
            fn_k0d_gather_zero = _k0d_mods["k0d_gather_zero"].get_function("k0d_gather_zero")
            fn_k0d_combine = _k0d_mods["k0d_combine"].get_function("k0d_combine")
            fn_k0d_push = _k0d_mods["k0d_push"].get_function("k0d_push")
            fn_k0d_dsort = _k0d_mods["k0d_dsort"].get_function("k0d_dsort")
            fn_k0d_mega = _k0d_mods["k0d_mega"].get_function("k0d_mega")
        if "k0d_phase_marker" in _k0d_mods:
            fn_k0d_phase_marker = _k0d_mods["k0d_phase_marker"].get_function("k0d_phase_marker")
            fn_k0d_phase_timer_calibration = _k0d_mods["k0d_phase_marker"].get_function(
                "k0d_phase_timer_calibration"
            )
        if "k0d_db_micro" in _k0d_mods:
            fn_k0d_db_micro = _k0d_mods["k0d_db_micro"].get_function("k0d_db_micro")
        if "k0d_mega_ts" in _k0d_mods:
            fn_k0d_mega_ts = _k0d_mods["k0d_mega_ts"].get_function("k0d_mega_ts")
        if K0D_REQUESTED:
            R["k0d"]["source_sha256"] = {}
            for _k0d_name in K0D_NAMES:
                _k0d_path = os.path.join(K0D_KERNEL_DIR, _k0d_name + ".hip")
                with open(_k0d_path, "rb") as _k0d_fh:
                    R["k0d"]["source_sha256"][_k0d_name] = hashlib.sha256(
                        _k0d_fh.read()
                    ).hexdigest()
            R["k0d"]["hsaco_sha256"] = {}
            R["k0d"]["hsaco_path"] = {}
            for _k0d_name in K0D_NAMES:
                _k0d_hsaco = _k0d_jit_loader._compiled_hsaco.get(_k0d_name)
                if _k0d_hsaco and os.path.isfile(_k0d_hsaco):
                    with open(_k0d_hsaco, "rb") as _k0d_fh:
                        R["k0d"]["hsaco_sha256"][_k0d_name] = hashlib.sha256(
                            _k0d_fh.read()
                        ).hexdigest()
                R["k0d"]["hsaco_path"][_k0d_name] = _k0d_hsaco
        _k0d_diag_loaded = [nm for nm in _K0D_JIT_NAMES if nm not in K0D_NAMES]
        if _k0d_diag_loaded:
            R["k0d_diag"] = dict(
                modules=list(_k0d_diag_loaded),
                standalone=(not K0D_REQUESTED),
                source_sha256={},
                hsaco_sha256={},
            )
            for _k0d_name in _k0d_diag_loaded:
                _k0d_path = os.path.join(K0D_KERNEL_DIR, _k0d_name + ".hip")
                with open(_k0d_path, "rb") as _k0d_fh:
                    R["k0d_diag"]["source_sha256"][_k0d_name] = hashlib.sha256(
                        _k0d_fh.read()
                    ).hexdigest()
                _k0d_hsaco = _k0d_jit_loader._compiled_hsaco.get(_k0d_name)
                if _k0d_hsaco and os.path.isfile(_k0d_hsaco):
                    with open(_k0d_hsaco, "rb") as _k0d_fh:
                        R["k0d_diag"]["hsaco_sha256"][_k0d_name] = hashlib.sha256(
                            _k0d_fh.read()
                        ).hexdigest()
        K0D_OK = True
    except Exception as _e_k0d:
        R["k0d"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_k0d), _e_k0d)
        ).strip()
    R["k0d"]["modules_loaded"] = K0D_OK and K0D_REQUESTED
    if not K0D_OK:
        raise RuntimeError(f"k0d JIT/load failed: {R['k0d'].get('load_error')}")

# ---- k0pf3 prefill package: source JIT + per-module shmem initialization. ----
PF3_OK = False
fn_pf3_qpush = fn_pf3_dsort = fn_pf3_mega = None
fn_pf4_dsort = None
fn_pf3_qpush_ts = fn_pf3_dsort_ts = None
fn_pf4_dsort_ts = None
_PF3_LOAD_NAMES = list(PF3_NAMES) + (
    list(PF3_DIAG_NAMES) if _K0_PF3_TS else []
) + (["k0pf4_dsort_ts"] if _K0_PF4_TS else [])
R["pf3"] = dict(
    requested=PF3_REQUESTED,
    requested_arms=list(PF3_REQUESTED_ARMS),
    prefill_only=True,
    modules_loaded=False,
    grids=dict(
        qpush=PF3_QPUSH_GRID, dsort=PF3_DSORT_GRID, mega=PF3_MEGA_GRID,
        pf4_dsort=PF4_DSORT_GRID,
    ),
)
if PF3_REQUESTED or (_K0_PF3_TS and _K0D_STANDALONE):
    _pf3_copy_err = [None]
    if rank == 0:
        try:
            if PF4H_REQUESTED:
                for _hkp_name in HKP_HEADERS:
                    _hkp_src = os.path.join(HKP_HEADER_DIR, _hkp_name)
                    if not os.path.isfile(_hkp_src):
                        raise FileNotFoundError(_hkp_src)
                    shutil.copyfile(_hkp_src, os.path.join(KERNELS_DIR, _hkp_name))
            for _pf3_name in _PF3_LOAD_NAMES:
                _pf3_src = os.path.join(PF3_KERNEL_DIR, _pf3_name + ".hip")
                if not os.path.isfile(_pf3_src):
                    raise FileNotFoundError(_pf3_src)
                shutil.copyfile(_pf3_src, os.path.join(KERNELS_DIR, _pf3_name + ".hip"))
        except Exception as _e_pf3_copy:
            _pf3_copy_err[0] = "".join(
                traceback.format_exception_only(type(_e_pf3_copy), _e_pf3_copy)
            ).strip()
    dist.broadcast_object_list(_pf3_copy_err, src=0)
    if _pf3_copy_err[0] is not None:
        raise RuntimeError(f"k0pf3 source install failed: {_pf3_copy_err[0]}")
    dist.barrier()
    try:
        for _pf3_name in _PF3_LOAD_NAMES:
            ensure_compiled(_pf3_name)
        _pf3_qpush_mod = load_hip_module("k0pf3_qpush", init_shmem=True)
        _pf3_dsort_mod = load_hip_module("k0pf3_dsort", init_shmem=True)
        _pf3_mega_mod = load_hip_module("k0pf3_mega", init_shmem=True)
        fn_pf3_qpush = _pf3_qpush_mod.get_function("k0pf3_qpush")
        fn_pf3_dsort = _pf3_dsort_mod.get_function("k0pf3_dsort")
        fn_pf3_mega = _pf3_mega_mod.get_function("k0pf3_mega")
        if "k0pf4_dsort" in _PF3_LOAD_NAMES:
            fn_pf4_dsort = load_hip_module("k0pf4_dsort", init_shmem=True).get_function(
                "k0pf4_dsort"
            )
            R["pf3"]["hkp_header_sha256"] = {}
            for _hkp_name in HKP_HEADERS:
                with open(os.path.join(HKP_HEADER_DIR, _hkp_name), "rb") as _hkp_fh:
                    R["pf3"]["hkp_header_sha256"][_hkp_name] = hashlib.sha256(
                        _hkp_fh.read()
                    ).hexdigest()
        if "k0pf3_qpush_ts" in _PF3_LOAD_NAMES:
            fn_pf3_qpush_ts = load_hip_module("k0pf3_qpush_ts", init_shmem=True).get_function(
                "k0pf3_qpush_ts"
            )
            fn_pf3_dsort_ts = load_hip_module("k0pf3_dsort_ts", init_shmem=True).get_function(
                "k0pf3_dsort_ts"
            )
            R["pf3"]["diag_source_sha256"] = {}
            if _K0_PF4_TS:
                fn_pf4_dsort_ts = load_hip_module(
                    "k0pf4_dsort_ts", init_shmem=True
                ).get_function("k0pf4_dsort_ts")
                _pf4ts_path = os.path.join(PF3_KERNEL_DIR, "k0pf4_dsort_ts.hip")
                with open(_pf4ts_path, "rb") as _pf4ts_fh:
                    R["pf3"]["diag_source_sha256"]["k0pf4_dsort_ts"] = hashlib.sha256(
                        _pf4ts_fh.read()
                    ).hexdigest()
            for _pf3_name in PF3_DIAG_NAMES:
                _pf3_path = os.path.join(PF3_KERNEL_DIR, _pf3_name + ".hip")
                with open(_pf3_path, "rb") as _pf3_fh:
                    R["pf3"]["diag_source_sha256"][_pf3_name] = hashlib.sha256(
                        _pf3_fh.read()
                    ).hexdigest()
        R["pf3"]["source_sha256"] = {}
        R["pf3"]["hsaco_sha256"] = {}
        R["pf3"]["hsaco_path"] = {}
        for _pf3_name in PF3_NAMES:
            _pf3_path = os.path.join(PF3_KERNEL_DIR, _pf3_name + ".hip")
            with open(_pf3_path, "rb") as _pf3_fh:
                R["pf3"]["source_sha256"][_pf3_name] = hashlib.sha256(
                    _pf3_fh.read()
                ).hexdigest()
            _pf3_hsaco = _k0d_jit_loader._compiled_hsaco.get(_pf3_name)
            if _pf3_hsaco and os.path.isfile(_pf3_hsaco):
                with open(_pf3_hsaco, "rb") as _pf3_fh:
                    R["pf3"]["hsaco_sha256"][_pf3_name] = hashlib.sha256(
                        _pf3_fh.read()
                    ).hexdigest()
                R["pf3"]["hsaco_path"][_pf3_name] = _pf3_hsaco
        # The source launch bound reserves at least four resident 256-thread CTAs/CU. Check the
        # runtime-reported CU count rather than baking a node-specific residency ceiling.
        _pf3_resident_floor = 4 * cu
        if PF3_MEGA_GRID > _pf3_resident_floor:
            raise RuntimeError(
                f"k0pf3_mega grid {PF3_MEGA_GRID} exceeds conservative residency floor "
                f"{_pf3_resident_floor} ({cu} CUs * 4 CTAs/CU)"
            )
        R["pf3"]["mega_residency"] = dict(
            grid=PF3_MEGA_GRID, cu=cu, conservative_ctas_per_cu=4,
            resident_floor=_pf3_resident_floor, pass_check=True,
        )
        # k0pf4_dsort spins on TWO internal grid barriers, so every CTA must be co-resident.
        if PF4H_REQUESTED:
            if PF4_DSORT_GRID > _pf3_resident_floor or PF4_DSORT_GRID > PF4_MAXCTA:
                raise RuntimeError(
                    f"k0pf4_dsort grid {PF4_DSORT_GRID} exceeds the residency floor "
                    f"{_pf3_resident_floor} or the kernel's K0P4_MAXCTA {PF4_MAXCTA}"
                )
            R["pf3"]["pf4_dsort_residency"] = dict(
                grid=PF4_DSORT_GRID, cu=cu, resident_floor=_pf3_resident_floor,
                max_cta=PF4_MAXCTA, pass_check=True,
            )
        PF3_OK = True
    except Exception as _e_pf3:
        R["pf3"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf3), _e_pf3)
        ).strip()
    R["pf3"]["modules_loaded"] = PF3_OK
    if not PF3_OK:
        raise RuntimeError(f"k0pf3 JIT/load failed: {R['pf3'].get('load_error')}")

# ---- k0pf5 LL128 doorbell-free dispatch pair. Source JIT, additive arm only. ----
PF5_OK = False
fn_pf5_qpush = fn_pf5_dsort = None
R["pf5"] = dict(
    requested=PF5_REQUESTED,
    requested_arms=list(PF5_REQUESTED_ARMS),
    modules_loaded=False,
    grids=dict(qpush=PF5_QPUSH_GRID, dsort=PF5_DSORT_GRID),
    row_stride=PF5_ROW_STRIDE,
)
if PF5_REQUESTED:
    _pf5_copy_err = [None]
    if rank == 0:
        try:
            for _header_name in HKP_HEADERS + PF5_HEADERS:
                _header_root = HKP_HEADER_DIR if _header_name in HKP_HEADERS else PF3_KERNEL_DIR
                _header_src = os.path.join(_header_root, _header_name)
                if not os.path.isfile(_header_src):
                    raise FileNotFoundError(_header_src)
                shutil.copyfile(_header_src, os.path.join(KERNELS_DIR, _header_name))
            for _pf5_name in PF5_NAMES:
                _pf5_src = os.path.join(PF3_KERNEL_DIR, _pf5_name + ".hip")
                if not os.path.isfile(_pf5_src):
                    raise FileNotFoundError(_pf5_src)
                shutil.copyfile(_pf5_src, os.path.join(KERNELS_DIR, _pf5_name + ".hip"))
        except Exception as _e_pf5_copy:
            _pf5_copy_err[0] = "".join(
                traceback.format_exception_only(type(_e_pf5_copy), _e_pf5_copy)
            ).strip()
    dist.broadcast_object_list(_pf5_copy_err, src=0)
    if _pf5_copy_err[0] is not None:
        raise RuntimeError(f"k0pf5 source install failed: {_pf5_copy_err[0]}")
    dist.barrier()
    try:
        for _pf5_name in PF5_NAMES:
            ensure_compiled(_pf5_name)
        _pf5_qmod = load_hip_module("k0pf5_qpush_ll", init_shmem=True)
        _pf5_dmod = load_hip_module("k0pf5_dsort_ll", init_shmem=False)
        fn_pf5_qpush = _pf5_qmod.get_function("k0pf5_qpush_ll")
        fn_pf5_dsort = _pf5_dmod.get_function("k0pf5_dsort_ll")
        R["pf5"]["source_sha256"] = {}
        R["pf5"]["hsaco_sha256"] = {}
        R["pf5"]["hsaco_path"] = {}
        for _pf5_name in PF5_NAMES:
            _pf5_path = os.path.join(PF3_KERNEL_DIR, _pf5_name + ".hip")
            with open(_pf5_path, "rb") as _pf5_fh:
                R["pf5"]["source_sha256"][_pf5_name] = hashlib.sha256(
                    _pf5_fh.read()
                ).hexdigest()
            _pf5_hsaco = _k0d_jit_loader._compiled_hsaco.get(_pf5_name)
            if _pf5_hsaco and os.path.isfile(_pf5_hsaco):
                with open(_pf5_hsaco, "rb") as _pf5_fh:
                    R["pf5"]["hsaco_sha256"][_pf5_name] = hashlib.sha256(
                        _pf5_fh.read()
                    ).hexdigest()
            R["pf5"]["hsaco_path"][_pf5_name] = _pf5_hsaco
        _pf5_resident_floor = 4 * cu
        if PF5_DSORT_GRID > _pf5_resident_floor or PF5_DSORT_GRID > PF4_MAXCTA:
            raise RuntimeError(
                f"k0pf5_dsort_ll grid {PF5_DSORT_GRID} exceeds residency floor "
                f"{_pf5_resident_floor} or max CTA {PF4_MAXCTA}"
            )
        R["pf5"]["dsort_residency"] = dict(
            grid=PF5_DSORT_GRID,
            cu=cu,
            resident_floor=_pf5_resident_floor,
            max_cta=PF4_MAXCTA,
            pass_check=True,
        )
        PF5_OK = True
    except Exception as _e_pf5:
        R["pf5"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf5), _e_pf5)
        ).strip()
    R["pf5"]["modules_loaded"] = PF5_OK
    if not PF5_OK:
        raise RuntimeError(f"k0pf5 JIT/load failed: {R['pf5'].get('load_error')}")

# ---- k0pf6 full-region megakernel. Source JIT, additive arm only. -----------------------
PF6_OK = False
fn_pf6_mega = None
_PF6_SOURCE_SPECS = (
    (PF6_NAME + ".hip", os.path.join(PF3_KERNEL_DIR, PF6_NAME + ".hip")),
    (PF6C_NAME + ".hip", os.path.join(PF3_KERNEL_DIR, PF6C_NAME + ".hip")),  # exp_56
    ("k0pf5_ll128.hpp", os.path.join(PF3_KERNEL_DIR, "k0pf5_ll128.hpp")),
    ("k0pf6_chunk_release.hpp",
     os.path.join(PF3_KERNEL_DIR, "k0pf6_chunk_release.hpp")),  # exp_56 additive header
) + (
    ((PF6GM_NAME + ".hip", os.path.join(PF3_KERNEL_DIR, PF6GM_NAME + ".hip")),)
    if PF6GM_REQUESTED else ()
) + (
    (
        (
            PF6MPS_NAME + ".hip",
            os.path.join(PF6MPS_SOURCE_DIR, "k0pf6gm_device_tile_mps.hip"),
        ),
        (
            "moe_hk_adapter.cuh",
            os.path.join(PF6MPS_SOURCE_DIR, "moe_hk_adapter.cuh"),
        ),
        (
            "moe_mps_adapter.cuh",
            os.path.join(PF6MPS_SOURCE_DIR, "moe_mps_adapter.cuh"),
        ),
        (
            PF6MPS_SNAPSHOT_NAME + ".hip",
            os.path.join(PF3_KERNEL_DIR, PF6MPS_SNAPSHOT_NAME + ".hip"),
        ),
    )
    if PF6MPS_REQUESTED else ()
) + tuple(
    (_header_name, os.path.join(HKP_HEADER_DIR, _header_name))
    for _header_name in HKP_HEADERS
) + tuple(
    (
        _source_name,
        os.path.join(
            (
                PF6MPS_SOURCE_DIR
                if _source_name in ("n2_phase1_gm_mps.cpp",
                                    "n2_phase2_gm_mps.cpp")
                else PF6_N2_SOURCE_DIR
            ),
            _source_name,
        ),
    )
    for _source_name in PF6_N2_FILES
)
R["pf6"] = dict(
    requested=PF6_REQUESTED,
    requested_arms=list(PF6_REQUESTED_ARMS),
    modules_loaded=False,
    grid=dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS),
    row_stride=PF5_ROW_STRIDE,
)
if PF6_REQUESTED:
    _pf6_copy_err = [None]
    if rank == 0:
        try:
            for _install_name, _source_path in _PF6_SOURCE_SPECS:
                if not os.path.isfile(_source_path):
                    raise FileNotFoundError(_source_path)
                shutil.copyfile(
                    _source_path, os.path.join(KERNELS_DIR, _install_name)
                )
        except Exception as _e_pf6_copy:
            _pf6_copy_err[0] = "".join(
                traceback.format_exception_only(type(_e_pf6_copy), _e_pf6_copy)
            ).strip()
    dist.broadcast_object_list(_pf6_copy_err, src=0)
    if _pf6_copy_err[0] is not None:
        raise RuntimeError(f"k0pf6 source install failed: {_pf6_copy_err[0]}")
    dist.barrier()
    try:
        # Mori's stock genco driver does not carry the HipKittens include roots or the
        # compile switches used by the verified N2 build. hipcc's documented append hook
        # keeps this one JIT compile source-based without mutating Mori or the authored kernel.
        _pf6_hk_include = os.path.join(PF6_HIPKITTENS_ROOT, "include")
        _pf6_hk_prototype = os.path.join(PF6_HIPKITTENS_ROOT, "prototype")
        if not os.path.isfile(os.path.join(_pf6_hk_include, "kittens.cuh")):
            raise FileNotFoundError(os.path.join(_pf6_hk_include, "kittens.cuh"))
        if any(" " in _path for _path in (_pf6_hk_include, _pf6_hk_prototype)):
            raise RuntimeError(
                "PF6 HipKittens include paths cannot contain spaces because hipcc "
                "parses HIPCC_COMPILE_FLAGS_APPEND as a flag string"
            )
        _pf6_compile_flags = " ".join(
            (
                "-std=c++20",
                "-O3",
                "-DKITTENS_CDNA4",
                "-ffast-math",
                "-mllvm",
                "-amdgpu-mfma-vgpr-form=1",
                f"-I{_pf6_hk_include}",
                f"-I{_pf6_hk_prototype}",
            )
        )
        _pf6_old_compile_flags = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(
            x for x in (_pf6_old_compile_flags, _pf6_compile_flags) if x
        )
        try:
            ensure_compiled(PF6_NAME)
        finally:
            if _pf6_old_compile_flags is None:
                os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
            else:
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = _pf6_old_compile_flags
        _pf6_mod = load_hip_module(PF6_NAME, init_shmem=True)
        fn_pf6_mega = _pf6_mod.get_function(PF6_NAME)
        R["pf6"]["jit_compile_flags"] = _pf6_compile_flags
        R["pf6"]["hipkittens_root"] = PF6_HIPKITTENS_ROOT
        R["pf6"]["source_path"] = {
            _install_name: _source_path
            for _install_name, _source_path in _PF6_SOURCE_SPECS
        }
        R["pf6"]["source_sha256"] = {}
        for _install_name, _source_path in _PF6_SOURCE_SPECS:
            with open(_source_path, "rb") as _pf6_source_fh:
                R["pf6"]["source_sha256"][_install_name] = hashlib.sha256(
                    _pf6_source_fh.read()
                ).hexdigest()
        _pf6_hsaco = _k0d_jit_loader._compiled_hsaco.get(PF6_NAME)
        if not _pf6_hsaco or not os.path.isfile(_pf6_hsaco):
            raise FileNotFoundError(
                f"k0pf6 compiled HSACO is unavailable: {_pf6_hsaco!r}"
            )
        with open(_pf6_hsaco, "rb") as _pf6_hsaco_fh:
            _pf6_hsaco_sha256 = hashlib.sha256(_pf6_hsaco_fh.read()).hexdigest()
        R["pf6"]["hsaco_path"] = {PF6_NAME: _pf6_hsaco}
        R["pf6"]["hsaco_sha256"] = {PF6_NAME: _pf6_hsaco_sha256}
        # The authored kernel uses counted whole-grid barriers and launch_bounds(256, 1).
        # Conservatively require one simultaneously resident CTA per CU.
        if PF6_MEGA_GRID > cu:
            raise RuntimeError(
                f"k0pf6_mega grid {PF6_MEGA_GRID} exceeds the conservative "
                f"one-CTA-per-CU residency floor {cu}"
            )
        R["pf6"]["mega_residency"] = dict(
            grid=PF6_MEGA_GRID,
            threads=PF6_THREADS,
            cu=cu,
            conservative_ctas_per_cu=1,
            resident_floor=cu,
            pass_check=True,
        )
        PF6_OK = True
    except Exception as _e_pf6:
        R["pf6"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf6), _e_pf6)
        ).strip()
    R["pf6"]["modules_loaded"] = PF6_OK
    if not PF6_OK:
        raise RuntimeError(f"k0pf6 JIT/load failed: {R['pf6'].get('load_error')}")

# ---- exp_56 k0pf6c chunk-counted-release megakernel. Same source-JIT machinery. ----------
PF6C_OK = False
fn_pf6c_mega = None
R["pf6c"] = dict(
    requested=PF6C_REQUESTED,
    requested_arms=list(PF6C_REQUESTED_ARMS),
    modules_loaded=False,
    grid=dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS),
)
if PF6C_REQUESTED:
    # sources were already installed into KERNELS_DIR by the shared _PF6_SOURCE_SPECS copy above.
    try:
        _pf6c_hk_include = os.path.join(PF6_HIPKITTENS_ROOT, "include")
        _pf6c_hk_prototype = os.path.join(PF6_HIPKITTENS_ROOT, "prototype")
        _pf6c_compile_flags = " ".join(
            (
                "-std=c++20", "-O3", "-DKITTENS_CDNA4", "-ffast-math",
                "-mllvm", "-amdgpu-mfma-vgpr-form=1",
                f"-I{_pf6c_hk_include}", f"-I{_pf6c_hk_prototype}",
            )
        )
        _pf6c_old = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(
            x for x in (_pf6c_old, _pf6c_compile_flags) if x
        )
        try:
            ensure_compiled(PF6C_NAME)
        finally:
            if _pf6c_old is None:
                os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
            else:
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = _pf6c_old
        _pf6c_mod = load_hip_module(PF6C_NAME, init_shmem=True)
        fn_pf6c_mega = _pf6c_mod.get_function(PF6C_NAME)
        R["pf6c"]["jit_compile_flags"] = _pf6c_compile_flags
        _pf6c_hsaco = _k0d_jit_loader._compiled_hsaco.get(PF6C_NAME)
        if not _pf6c_hsaco or not os.path.isfile(_pf6c_hsaco):
            raise FileNotFoundError(f"k0pf6c compiled HSACO is unavailable: {_pf6c_hsaco!r}")
        with open(_pf6c_hsaco, "rb") as _fh:
            R["pf6c"]["hsaco_sha256"] = {PF6C_NAME: hashlib.sha256(_fh.read()).hexdigest()}
        R["pf6c"]["hsaco_path"] = {PF6C_NAME: _pf6c_hsaco}
        with open(os.path.join(PF3_KERNEL_DIR, PF6C_NAME + ".hip"), "rb") as _fh:
            R["pf6c"]["source_sha256_hip"] = hashlib.sha256(_fh.read()).hexdigest()
        with open(os.path.join(PF3_KERNEL_DIR, "k0pf6_chunk_release.hpp"), "rb") as _fh:
            R["pf6c"]["source_sha256_hdr"] = hashlib.sha256(_fh.read()).hexdigest()
        if PF6_MEGA_GRID > cu:
            raise RuntimeError(
                f"k0pf6c_mega grid {PF6_MEGA_GRID} exceeds one-CTA-per-CU floor {cu}"
            )
        PF6C_OK = True
    except Exception as _e_pf6c:
        R["pf6c"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf6c), _e_pf6c)
        ).strip()
    R["pf6c"]["modules_loaded"] = PF6C_OK
    if not PF6C_OK:
        raise RuntimeError(f"k0pf6c JIT/load failed: {R['pf6c'].get('load_error')}")

# ---- exp_59 k0pf6gm large-M G-stacked megakernel. Additive arm only. ----------------------
PF6GM_OK = False
fn_pf6gm_mega = None
R["pf6gm"] = dict(
    requested=PF6GM_REQUESTED,
    requested_arms=list(PF6GM_REQUESTED_ARMS),
    modules_loaded=False,
    grid=dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS),
    gstack_G=PF6GM_G,
)
if PF6GM_REQUESTED:
    # sources were already installed into KERNELS_DIR by the shared _PF6_SOURCE_SPECS copy above.
    try:
        _pf6gm_hk_include = os.path.join(PF6_HIPKITTENS_ROOT, "include")
        _pf6gm_hk_prototype = os.path.join(PF6_HIPKITTENS_ROOT, "prototype")
        # -DK0P6GM_G / -DN2GM_G MUST be equal so the mega tile-table G matches the phase bodies.
        _pf6gm_compile_flags = " ".join(
            (
                "-std=c++20", "-O3", "-DKITTENS_CDNA4", "-ffast-math",
                "-mllvm", "-amdgpu-mfma-vgpr-form=1",
                f"-DK0P6GM_G={PF6GM_G}", f"-DN2GM_G={PF6GM_G}",
                f"-I{_pf6gm_hk_include}", f"-I{_pf6gm_hk_prototype}",
            )
        )
        _pf6gm_old = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(
            x for x in (_pf6gm_old, _pf6gm_compile_flags) if x
        )
        try:
            ensure_compiled(PF6GM_NAME)
        finally:
            if _pf6gm_old is None:
                os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
            else:
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = _pf6gm_old
        _pf6gm_mod = load_hip_module(PF6GM_NAME, init_shmem=True)
        fn_pf6gm_mega = _pf6gm_mod.get_function(PF6GM_NAME)
        R["pf6gm"]["jit_compile_flags"] = _pf6gm_compile_flags
        _pf6gm_hsaco = _k0d_jit_loader._compiled_hsaco.get(PF6GM_NAME)
        if not _pf6gm_hsaco or not os.path.isfile(_pf6gm_hsaco):
            raise FileNotFoundError(f"k0pf6gm compiled HSACO is unavailable: {_pf6gm_hsaco!r}")
        with open(_pf6gm_hsaco, "rb") as _fh:
            R["pf6gm"]["hsaco_sha256"] = {PF6GM_NAME: hashlib.sha256(_fh.read()).hexdigest()}
        R["pf6gm"]["hsaco_path"] = {PF6GM_NAME: _pf6gm_hsaco}
        R["pf6gm"]["source_sha256"] = {}
        for _sn in (PF6GM_NAME + ".hip", "n2_phase1_gm.cpp", "n2_phase2_gm.cpp"):
            _sp = os.path.join(PF3_KERNEL_DIR if _sn.endswith(".hip") else PF6_N2_SOURCE_DIR, _sn)
            with open(_sp, "rb") as _fh:
                R["pf6gm"]["source_sha256"][_sn] = hashlib.sha256(_fh.read()).hexdigest()
        if PF6_MEGA_GRID > cu:
            raise RuntimeError(
                f"k0pf6gm_mega grid {PF6_MEGA_GRID} exceeds one-CTA-per-CU floor {cu}"
            )
        PF6GM_OK = True
    except Exception as _e_pf6gm:
        R["pf6gm"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf6gm), _e_pf6gm)
        ).strip()
    R["pf6gm"]["modules_loaded"] = PF6GM_OK
    if not PF6GM_OK:
        raise RuntimeError(f"k0pf6gm JIT/load failed: {R['pf6gm'].get('load_error')}")

# ---- minimum-progress specialization sibling. Distinct source/cache key. ----------------
PF6MPS_OK = False
fn_pf6mps_mega = None
fn_pf6mps_heap_snapshot = None
R["pf6mps"] = dict(
    requested=PF6MPS_REQUESTED,
    requested_arms=list(PF6MPS_REQUESTED_ARMS),
    modules_loaded=False,
    grid=dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS),
    gstack_G=PF6MPS_G,
    config=PF6MPS_CONFIG,
)
if PF6MPS_REQUESTED:
    try:
        _pf6mps_hk_include = os.path.join(PF6MPS_DHK_ROOT, "include")
        _pf6mps_source_include = PF6MPS_SOURCE_DIR
        if not os.path.isfile(os.path.join(_pf6mps_hk_include, "kittens.cuh")):
            raise FileNotFoundError(os.path.join(_pf6mps_hk_include, "kittens.cuh"))
        if any(" " in path for path in (_pf6mps_hk_include, _pf6mps_source_include)):
            raise RuntimeError("MPS include paths cannot contain spaces")
        _pf6mps_effective_g = {
            "PF6GM_G": PF6MPS_G,
            "K0P6GM_G": PF6MPS_G,
            "N2GM_G": PF6MPS_G,
        }
        if len(set(_pf6mps_effective_g.values())) != 1:
            raise RuntimeError(f"MPS G-stack mismatch: {_pf6mps_effective_g}")
        _pf6mps_compile_flags = " ".join(
            (
                "-std=c++20", "-O3", "-DKITTENS_CDNA4",
                "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
                "-mllvm", "-amdgpu-mfma-vgpr-form=1",
                f"-DK0P6GM_G={PF6MPS_G}", f"-DN2GM_G={PF6MPS_G}",
                f"-I{_pf6mps_hk_include}", f"-I{_pf6mps_source_include}",
            )
        )
        _pf6mps_old = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(
            item for item in (_pf6mps_old, _pf6mps_compile_flags) if item
        )
        try:
            ensure_compiled(PF6MPS_SNAPSHOT_NAME)
            ensure_compiled(PF6MPS_NAME)
        finally:
            if _pf6mps_old is None:
                os.environ.pop("HIPCC_COMPILE_FLAGS_APPEND", None)
            else:
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = _pf6mps_old
        _pf6mps_snapshot_mod = load_hip_module(
            PF6MPS_SNAPSHOT_NAME, init_shmem=True
        )
        fn_pf6mps_heap_snapshot = _pf6mps_snapshot_mod.get_function(
            PF6MPS_SNAPSHOT_NAME
        )
        _pf6mps_mod = load_hip_module(PF6MPS_NAME, init_shmem=True)
        fn_pf6mps_mega = _pf6mps_mod.get_function(PF6MPS_NAME)
        R["pf6mps"]["jit_compile_flags"] = _pf6mps_compile_flags
        R["pf6mps"]["effective_g"] = _pf6mps_effective_g
        R["pf6mps"]["source_path"] = {
            install_name: source_path
            for install_name, source_path in _PF6_SOURCE_SPECS
            if install_name in (
                PF6MPS_NAME + ".hip",
                "moe_hk_adapter.cuh",
                "moe_mps_adapter.cuh",
                "n2_phase1_gm_mps.cpp",
                "n2_phase2_gm_mps.cpp",
                PF6MPS_SNAPSHOT_NAME + ".hip",
            )
        }
        R["pf6mps"]["source_sha256"] = {}
        for install_name, source_path in R["pf6mps"]["source_path"].items():
            with open(source_path, "rb") as source_file:
                R["pf6mps"]["source_sha256"][install_name] = hashlib.sha256(
                    source_file.read()
                ).hexdigest()
        R["pf6mps"]["hsaco_path"] = {}
        R["pf6mps"]["hsaco_sha256"] = {}
        for module_name in (PF6MPS_NAME, PF6MPS_SNAPSHOT_NAME):
            hsaco_path = _k0d_jit_loader._compiled_hsaco.get(module_name)
            if not hsaco_path or not os.path.isfile(hsaco_path):
                raise FileNotFoundError(
                    f"{module_name} compiled HSACO is unavailable: {hsaco_path!r}"
                )
            R["pf6mps"]["hsaco_path"][module_name] = hsaco_path
            with open(hsaco_path, "rb") as hsaco_file:
                R["pf6mps"]["hsaco_sha256"][module_name] = hashlib.sha256(
                    hsaco_file.read()
                ).hexdigest()
        if PF6_MEGA_GRID > cu:
            raise RuntimeError(
                f"{PF6MPS_NAME} grid {PF6_MEGA_GRID} exceeds one-CTA-per-CU floor {cu}"
            )
        PF6MPS_OK = True
    except Exception as _e_pf6mps:
        R["pf6mps"]["load_error"] = "".join(
            traceback.format_exception_only(type(_e_pf6mps), _e_pf6mps)
        ).strip()
    R["pf6mps"]["modules_loaded"] = PF6MPS_OK
    if not PF6MPS_OK:
        raise RuntimeError(
            f"MPS JIT/load failed: {R['pf6mps'].get('load_error')}"
        )

# ---- k0pf modules. Prefer preserved hsaco, but self-host from source when absent. ----
from mori.ops import _jit_loader as _pf_jit
_PF_SOURCE_DIR = os.environ.get(
    "K0_PF_SOURCE_DIR", os.path.join(_K0_ROOT, "prefill_opt", "kernels")
)
_pf_module_origin = {}
for _pf_name in ("k0pf_gather", "k0pf_combine", "k0pf_quant"):
    _pf_path = os.path.join(PF_HSACO_DIR, _pf_name + ".hsaco")
    if os.path.isfile(_pf_path):
        _pf_jit._compiled_hsaco[_pf_name] = _pf_path
        _pf_module_origin[_pf_name] = "preserved_hsaco"
    else:
        _pf_src = os.path.join(_PF_SOURCE_DIR, _pf_name + ".hip")
        if not os.path.isfile(_pf_src):
            raise FileNotFoundError(
                f"neither preserved hsaco {_pf_path} nor source {_pf_src} exists"
            )
        if rank == 0:
            shutil.copyfile(_pf_src, os.path.join(KERNELS_DIR, _pf_name + ".hip"))
        _pf_module_origin[_pf_name] = "source_jit"
dist.barrier()
for _pf_name, _pf_origin in _pf_module_origin.items():
    if _pf_origin == "source_jit":
        ensure_compiled(_pf_name)
mod_g_pf = load_hip_module("k0pf_gather", init_shmem=True)
mod_c_pf = load_hip_module("k0pf_combine", init_shmem=True)
mod_q_pf = load_hip_module("k0pf_quant", init_shmem=True)
fn_gather_pf = mod_g_pf.get_function("k0pf_gather_rows_fp8")
fn_combine_pf = mod_c_pf.get_function("k0pf_combine")
fn_quant_pf = mod_q_pf.get_function("k0pf_quant_to_shmem")
R["k0pf_modules"] = {
    "origin": _pf_module_origin,
    "hsaco_path": {
        name: _pf_jit._compiled_hsaco.get(name) for name in _pf_module_origin
    },
}

# ---- C-3: multi-token combine (exp_51's depth lever) -------------------------------------------
# K0_COMBINE_ARM=mtab2|mtab4 JIT-compiles k0pf_combine_mtab.hip and rebinds fn_combine_pf to it.
# That file is the A/B-ABI twin of k0pf_combine_multitok.hip: identical depth mechanism, but the
# frozen 8-kernarg e22 shape and ShmemPtrP2p translation instead of the deployed isolation-v1
# 9-kernarg + SymmMemTopologyV1 shape. Launching the 9-arg kernel from this 8-arg site would read
# past the kernarg buffer (§L90's SIGSEGV-on-6-of-8-ranks failure), so the ABI twin is required.
# Rebinding ONE symbol means every call site, gate, timing arm, and stress replay below picks the
# arm up unchanged -- the combine becomes the single variable against the same captured route.
_K0_COMBINE_ARM = os.environ.get("K0_COMBINE_ARM", "").strip()
R["combine_arm"] = _K0_COMBINE_ARM or "k0pf_combine"
if _K0_COMBINE_ARM:
    _MTAB_SYMS = {"mtab2": "k0pf_combine_mtab2", "mtab4": "k0pf_combine_mtab4"}
    if _K0_COMBINE_ARM not in _MTAB_SYMS:
        raise RuntimeError(
            f"K0_COMBINE_ARM={_K0_COMBINE_ARM!r} unknown; choose one of {sorted(_MTAB_SYMS)}. "
            "An unrecognized arm must be an ERROR, never a silent no-op (§L90)."
        )
    _mtab_src = os.path.join(PF3_KERNEL_DIR, "k0pf_combine_mtab.hip")
    if rank == 0:
        if not os.path.isfile(_mtab_src):
            raise FileNotFoundError(_mtab_src)
        shutil.copyfile(_mtab_src, os.path.join(KERNELS_DIR, "k0pf_combine_mtab.hip"))
    dist.barrier()
    ensure_compiled("k0pf_combine_mtab")
    _mod_mtab = load_hip_module("k0pf_combine_mtab", init_shmem=True)
    fn_combine_pf = _mod_mtab.get_function(_MTAB_SYMS[_K0_COMBINE_ARM])
    with open(_mtab_src, "rb") as _mtab_fh:
        R["combine_arm_sha256"] = hashlib.sha256(_mtab_fh.read()).hexdigest()
    if rank == 0:
        print(f"[K0 COMBINE ARM] {_MTAB_SYMS[_K0_COMBINE_ARM]} "
              f"(sha256 {R['combine_arm_sha256'][:12]})", flush=True)

# ---- k0pf2 source-push dispatch. The preserved hsaco still goes through load_hip_module so this
# separately-loaded module receives its own shmem global-state initialization.
_PF2_PUSH_OK = False
fn_push_pf2 = fn_push_pf2_fused = fn_sc_transpose_pf2 = fn_sc_transpose_wait_pf2 = None
try:
    if os.path.isfile(PF2_HSACO):
        _pf_jit._compiled_hsaco["k0pf2_push"] = PF2_HSACO
        mod_push_pf2 = load_hip_module("k0pf2_push", init_shmem=True)
        fn_push_pf2 = mod_push_pf2.get_function("k0pf2_push")
        fn_push_pf2_fused = mod_push_pf2.get_function("k0pf2_push_fused")
        fn_sc_transpose_pf2 = mod_push_pf2.get_function("k0pf2_sc_transpose")
        fn_sc_transpose_wait_pf2 = mod_push_pf2.get_function("k0pf2_sc_transpose_wait")
        _PF2_PUSH_OK = True
except Exception as _e_pf2_push:
    R["pf2_push_load_err"] = "".join(
        traceback.format_exception_only(type(_e_pf2_push), _e_pf2_push)
    ).strip()
R["pf2_push_loaded"] = _PF2_PUSH_OK
R["pf2_plan_loaded"] = _HAS_PF2_PLAN
R["pf2_plan_import_err"] = _PF2_PLAN_ERR

# ---- N2R (exp_47-r): prebuilt hsaco with the folded rendezvous (GEMM tail fold + combine wait).
_N2R_OK = False
fn_n2r_p2 = fn_combine_wait = None
try:
    from mori.ops import _jit_loader
    _N2R_HSACO = os.environ.get("K0_N2R_HSACO", "/data/ecs-d2-ab/hsaco/k0_region_kernels_n2r.hsaco")
    if os.path.exists(_N2R_HSACO):
        # §L66's launcher trick: pre-populate the JIT cache so load_hip_module takes OUR hsaco
        # and STILL runs shmem_module_init (mandatory: the kernels call ShmemPtrP2p/ShmemGetMemBlock).
        _jit_loader._compiled_hsaco["k0_region_kernels_n2r"] = _N2R_HSACO
        n2r_mod = load_hip_module("k0_region_kernels_n2r", init_shmem=True)
        fn_n2r_p2 = n2r_mod.get_function("n2r_phase2_kernel")
        fn_combine_wait = n2r_mod.get_function("k0_region_combine_wait")
        _N2R_OK = True
except Exception as _e_n2r:
    R["n2r_load_err"] = "".join(traceback.format_exception_only(type(_e_n2r), _e_n2r)).strip()
R["n2r_loaded"] = _N2R_OK

# ---- PUSH: load the exp_001 push kernels if present; else the push arms are STUBBED. ----
PUSH_OK = False
fn_emask = fn_epoch = fn_dpush = fn_drel = fn_dacq = fn_dcopy = None
fn_cpub = fn_crel = fn_cacq = fn_reduce = None
try:
    if os.path.exists(PUSH_SRC):
        if rank == 0:
            # copy the aggregator AND its 3 #included sub-files into the MORI kernels dir so the
            # relative includes (k0_tile_push_kernels.hip -> expected_mask/dispatch_push/combine_push) resolve.
            _psrc_dir = os.path.dirname(PUSH_SRC)
            for _f in ("k0_tile_expected_mask.hip", "k0_tile_dispatch_push.hip", "k0_tile_combine_push.hip", PNAME + ".hip"):
                shutil.copyfile(os.path.join(_psrc_dir, _f), os.path.join(KERNELS_DIR, _f))
        dist.barrier()
        ensure_compiled(PNAME)
        # init_shmem=True: separately-loaded HIP modules each carry their OWN device globalGpuStates
        # (heapBaseAddr / p2pPeerPtrs); ShmemPtrP2p reads the CALLING module's state. Without initialising
        # the push module's state, ShmemPtrP2p returns 0 -> the addr-0 memory fault seen in dispatch_push.
        pmod = load_hip_module(PNAME, init_shmem=True)
        fn_emask  = pmod.get_function("k0_tile_expected_mask")     # own[] -> emask_disp[cur], emask_comb[cur]
        fn_epoch  = pmod.get_function("k0_tile_epoch_bump")        # gen[0] += 1  (device generation, no host readback)
        fn_dpush  = pmod.get_function("k0_tile_dispatch_push")     # ShmemPtrP2p direct stores -> a_push[slot]/sc_push[slot]; per-block system fence
        fn_drel   = pmod.get_function("k0_tile_dispatch_release")  # per-dest ShmemUint32AtomicAddThread(disp_flag+cur, 1, d)
        fn_dacq   = pmod.get_function("k0_tile_dispatch_acquire")  # spin disp_flag[emask_disp]>=gen (bounded/fail-closed); system fence
        fn_dcopy  = pmod.get_function("k0_tile_dispatch_copy")     # copy a_push[gen&1]/sc_push[gen&1] -> n1g's FIXED a_dst/sc_dst
        fn_cpub   = pmod.get_function("k0_tile_combine_publish")   # ShmemPtrP2p direct stores -> owner_slots[slot][tok][p]; per-block system fence
        fn_crel   = pmod.get_function("k0_tile_combine_release")   # per-owner ShmemUint32AtomicAddThread(comb_flag+cur, 1, owner)
        fn_cacq   = pmod.get_function("k0_tile_combine_acquire")   # spin comb_flag[emask_comb]>=gen (bounded/fail-closed); system fence
        fn_reduce = pmod.get_function("k0_tile_reduce_local")      # sum pull_src slots[gen&1] fp32 in pull_src order -> cand_out bf16 (RNE)
        PUSH_OK = True
except Exception as e:
    R["push_load_err"] = "".join(traceback.format_exception_only(type(e), e)).strip()
R["push_kernels_loaded"] = PUSH_OK

_ITEMS = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "uint8": 1}
_TS = {"bfloat16": "<u2", "float32": "<f4", "int32": "<i4", "int64": "<i8", "uint8": "|u1"}
_TD = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8}
_ka = []
def _view(ptr, shape, ts, td):
    class Wc:
        def __init__(s): s.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": ts, "data": (ptr, False), "version": 3, "strides": None}
    return torch.as_tensor(Wc(), device="cuda").view(td).view(*shape)
def mori_t(shape, dtype):
    n = 1
    for s in shape: n *= s
    ptr = ms.shmem_malloc(n * _ITEMS[dtype]); _ka.append(ptr); return _view(ptr, shape, _TS[dtype], _TD[dtype]), ptr
def mori_fp8(M, Kd):
    ptr = ms.shmem_malloc(M * Kd); _ka.append(ptr)
    return (_view(ptr, (M, Kd // 2), "<u2", torch.bfloat16), _view(ptr, (M, Kd), "|u1", torch.uint8), ptr)

# ---- symmetric source/staging buffers (frozen) ----
my_ids, my_ids_p = mori_t((T * TOPK, 1), "int32")
my_wgt, my_wgt_p = mori_t((T * TOPK, 1), "float32")
a_src_bf16, a_src_u8, a_src_p = mori_fp8(T, K)
sc_src, sc_src_p = mori_t((T, NG), "float32")
part, part_p = mori_t((MROWS, H), "bfloat16")
# k0pf2 destination is symmetric because origin ranks write it directly through ShmemPtrP2p.
# Scales land row-major in symmetric scratch, then transpose locally into n2's group-major ABI.
a_dst_pf2, a_dst_pf2_u8, a_dst_pf2_p = mori_fp8(MROWS, K)
sc_stage_pf2, sc_stage_pf2_p = mori_t((T_LOC_MAX, NG), "float32")
# PF3 arms get disjoint symmetric receive buffers and monotonic protocol state so capturing or
# replaying one graph cannot advance the other arm's epochs.
pf3_state = {}
if PF3_REQUESTED:
    for _pf3_arm in PF3_REQUESTED_ARMS:
        _pf3_a, _pf3_au8, _pf3_ap = mori_fp8(MROWS, K)
        _pf3_sc, _pf3_scp = mori_t((T_LOC_MAX, NG), "float32")
        _pf3_eid, _pf3_eidp = mori_t((T_LOC_MAX, TOPK), "int32")
        _pf3_wgt, _pf3_wgtp = mori_t((T_LOC_MAX, TOPK), "float32")
        _pf3_ctr, _pf3_ctrp = mori_t((1, 1), "int32")
        _pf3_db, _pf3_dbp = mori_t((WORLD, 1), "int32")
        _pf3_ctr.zero_()
        _pf3_db.zero_()
        pf3_state[_pf3_arm] = dict(
            a_dst=_pf3_a, a_dst_u8=_pf3_au8, a_dst_p=_pf3_ap,
            sc_stage=_pf3_sc, sc_stage_p=_pf3_scp,
            recv_eid=_pf3_eid, recv_eid_p=_pf3_eidp,
            recv_wgt=_pf3_wgt, recv_wgt_p=_pf3_wgtp,
            dest_counter=_pf3_ctr, dest_counter_p=_pf3_ctrp,
            doorbell=_pf3_db, doorbell_p=_pf3_dbp,
        )
# PF5 publishes LL128 rows into symmetric storage, then unpacks them locally in dsort.
pf5_state = {}
if PF5_REQUESTED:
    _pf5_ll, _pf5_llp = mori_t((T_LOC_MAX, PF5_ROW_STRIDE), "uint8")
    _pf5_rows_done, _pf5_rows_donep = mori_t((WORLD, 1), "int64")
    _pf5_ctr, _pf5_ctrp = mori_t((1, 1), "int32")
    _pf5_ll.zero_()
    _pf5_rows_done.zero_()
    _pf5_ctr.zero_()
    pf5_state = dict(
        a_ll=_pf5_ll,
        a_ll_p=_pf5_llp,
        rows_done=_pf5_rows_done,
        rows_done_p=_pf5_rows_donep,
        dest_counter=_pf5_ctr,
        dest_counter_p=_pf5_ctrp,
    )
# PF6 owns the complete region, so none of its symmetric epoch or payload storage aliases PF5.
pf6_state = {}
if PF6_REQUESTED:
    _pf6_ll, _pf6_llp = mori_t((T_LOC_MAX, PF5_ROW_STRIDE), "uint8")
    _pf6_rows_done, _pf6_rows_donep = mori_t((WORLD, 1), "int64")
    # exp_56 Option C + fix(5): dest_counter is EPOCH-PARITY DOUBLE-BUFFERED per-source:
    # dest_counter[2][world], indexed [epoch&1][src]. pf6_mega (dense) uses only slot 0, so this is
    # backward-compatible; pf6c_mega uses both parities. Zero-init means epoch-1 (parity 1) starts
    # clean and epoch-0 parity is never used first. chunk_ready is the per-(source,chunk) release
    # flag array, dest-side, symmetric (peers write it system-scope).
    _pf6_ctr, _pf6_ctrp = mori_t((2 * WORLD, 1), "int32")
    _PF6C_NCHUNK_MAX = 8   # MIRRORS K0P6C_NCHUNK_MAX in k0pf6c_mega.hip
    # exp_56 option b: self-describing 64-bit chunk words (epoch<<32 | fill_count). int64.
    _pf6_chunk_ready, _pf6_chunk_readyp = mori_t((WORLD, _PF6C_NCHUNK_MAX), "int64")
    # exp_56 diag: spin_dbg[0]=max spins to SUCCESS, [1]=max spins at FAIL (chunk poll). Local.
    _pf6_spin_dbg = torch.zeros((2,), dtype=torch.int32, device="cuda")
    # exp_59: expert-aligned tile table (device-written in M4). tile_desc holds one packed
    # (b0<<4)|gcount per tile; capacity is the max possible tile count = PADMAX/32 32-blocks
    # (upper bound at G=1). num_tiles is the device-written count. Local (not symmetric).
    _pf6_tile_desc = torch.zeros((PADMAX // 32,), dtype=torch.int32, device="cuda")
    # int[2]: [0]=device-written tile count, [1]=COMPILED K0P6GM_G (host asserts == PF6GM_G).
    _pf6_num_tiles = torch.zeros((2,), dtype=torch.int32, device="cuda")
    _pf6_part, _pf6_partp = mori_t((MROWS, H), "bfloat16")
    _pf6_maxb = PADMAX // 32
    # Per-producer, per-receive-row completion. Row ids are independent on each producer,
    # so the producer dimension prevents one rank's row r from releasing another rank's row r.
    # Epochs are monotonic; this array never needs clearing.
    _pf6_row_ready, _pf6_row_readyp = mori_t((TBO16_G * WORLD, T_LOC_MAX), "int32")
    _pf6_retired, _pf6_retiredp = mori_t((WORLD, 1), "int64")
    if PF6MPS_REQUESTED:
        _pf6_mps_t_ext = WORLD * MAXTOK
        _pf6_mps_queue = torch.zeros(
            (PADMAX // 32, 16), dtype=torch.int32, device="cuda"
        )
        _pf6_mps_arrivals = torch.zeros(
            (_pf6_mps_t_ext, 16), dtype=torch.int32, device="cuda"
        )
        _pf6_mps_pushed = torch.zeros(
            (_pf6_mps_t_ext,), dtype=torch.int32, device="cuda"
        )
        _pf6_mps_claim = torch.zeros(
            (_pf6_mps_t_ext,), dtype=torch.int32, device="cuda"
        )
        # 8 u32 words followed by 8 u64 timestamps = 96 bytes.
        _pf6_mps_state = torch.zeros((12,), dtype=torch.int64, device="cuda")
        _pf6_mps_slots, _pf6_mps_slotsp = mori_t(
            (TBO16_G * WORLD, MAXTOK, H), "bfloat16"
        )
        _pf6_mps_slots.zero_()
        # Snapshot the exact MoRI heap base and P2P peer heap bases from an
        # init_shmem=True module. The 9-word tensor is the device-visible
        # symmetric_heap_descriptor consumed from descriptor slot 55.
        _pf6_mps_heap_descriptor = torch.zeros(
            (9,), dtype=torch.int64, device="cuda"
        )
        fn_pf6mps_heap_snapshot.launch(
            (1,), (1,), 0, torch.cuda.current_stream().cuda_stream,
            _pf6_mps_heap_descriptor.data_ptr(),
        )
        torch.cuda.synchronize()
        _pf6_mps_heap_words = [
            int(word) for word in _pf6_mps_heap_descriptor.detach().cpu().tolist()
        ]
        if len(_pf6_mps_heap_words) != 9 or any(
            word == 0 for word in _pf6_mps_heap_words
        ):
            raise RuntimeError(
                f"invalid MoRI MPS heap descriptor: {_pf6_mps_heap_words}"
            )
        if _pf6_mps_heap_words[1 + rank] != _pf6_mps_heap_words[0]:
            raise RuntimeError(
                "MPS local heap base does not match the current-rank peer base"
            )
        R["pf6mps"]["heap_descriptor"] = {
            "local_heap_base": _pf6_mps_heap_words[0],
            "peer_heap_bases": _pf6_mps_heap_words[1:],
            "device_address": _pf6_mps_heap_descriptor.data_ptr(),
            "source": "MoRI module GpuStates heapBaseAddr/p2pPeerPtrs",
        }
        R["pf6mps"]["buffer_bytes"] = {
            "queue": (PADMAX // 32) * 16 * 4,
            "arrivals": _pf6_mps_t_ext * 16 * 4,
            "pushed": _pf6_mps_t_ext * 4,
            "claim": _pf6_mps_t_ext * 4,
            "state": 96,
            "slots": TBO16_G * WORLD * MAXTOK * H * 2,
        }
    for _pf6_symm_tensor in (
        _pf6_ll,
        _pf6_rows_done,
        _pf6_ctr,
        _pf6_part,
        _pf6_row_ready,
        _pf6_retired,
        _pf6_chunk_ready,
        *([_pf6_mps_slots] if PF6MPS_REQUESTED else []),
    ):
        _pf6_symm_tensor.zero_()
    pf6_state = dict(
        a_ll=_pf6_ll,
        a_ll_p=_pf6_llp,
        rows_done=_pf6_rows_done,
        rows_done_p=_pf6_rows_donep,
        dest_counter=_pf6_ctr,
        dest_counter_p=_pf6_ctrp,
        part=_pf6_part,
        part_p=_pf6_partp,
        row_ready=_pf6_row_ready,
        row_ready_p=_pf6_row_readyp,
        retired=_pf6_retired,
        retired_p=_pf6_retiredp,
        chunk_ready=_pf6_chunk_ready,
        chunk_ready_p=_pf6_chunk_readyp,
        spin_dbg=_pf6_spin_dbg,
        maxb=_pf6_maxb,
        tile_desc=_pf6_tile_desc,
        num_tiles=_pf6_num_tiles,
    )
    if PF6MPS_REQUESTED:
        pf6_state.update(
            mps_queue=_pf6_mps_queue,
            mps_arrivals=_pf6_mps_arrivals,
            mps_pushed=_pf6_mps_pushed,
            mps_claim=_pf6_mps_claim,
            mps_state=_pf6_mps_state,
            mps_slots=_pf6_mps_slots,
            mps_slots_p=_pf6_mps_slotsp,
            mps_heap_descriptor=_pf6_mps_heap_descriptor,
        )
# k0d push ingress landing/state.  Both push arms may share this monotonic state because the
# harness replays exactly one complete ingress->combine graph at a time in the same rank order.
a_dst_pd, a_dst_pd_u8, a_dst_pd_p = mori_fp8(MROWS, K)
sc_stage_pd, sc_stage_pd_p = mori_t((T_LOC_MAX, NG), "float32")
recv_eid_pd, recv_eid_pd_p = mori_t((T_LOC_MAX, TOPK), "int32")
recv_wgt_pd, recv_wgt_pd_p = mori_t((T_LOC_MAX, TOPK), "float32")
dest_counter_pd, dest_counter_pd_p = mori_t((1, 1), "int32")
recv_doorbell_pd, recv_doorbell_pd_p = mori_t((WORLD, 1), "int32")
dest_counter_pd.zero_()
recv_doorbell_pd.zero_()

# ---- PUSH: symmetric double-buffered payload landing buffers + uint32 flags + device gen/mask/err ----
# a_push[slot]/sc_push[slot] are the destination-final push targets (peers write them); the dispatch
# acquire copies slot gen&1 into n1g's FIXED local a_dst/sc_dst (n1g is unmodified — constraint #3).
# owner_slots[slot][owner_tok][producer][H] are the combine push targets (producer-keyed, collision-free).
a_push,  a_push_p  = mori_t((NSLOT, T_LOC_MAX, K), "uint8")            # fp8 rows, double-buffered
sc_push, sc_push_p = mori_t((NSLOT, NG * T_LOC_MAX, 1), "float32")     # group-major scales, double-buffered
oslots,  oslots_p  = mori_t((NSLOT, T, WORLD, H), "float32")           # combine slots, double-buffered (fp32)
# monotonic uint32 flag counters — allocated PER push-arm (below) so replaying one arm never perturbs another.
ms.shmem_barrier_all()

dev = "cuda"
all_ids = torch.zeros((WORLD * T * TOPK, 1), dtype=torch.int32, device="cuda")
all_wgt = torch.zeros((WORLD * T * TOPK, 1), dtype=torch.float32, device=dev)
N_GLOB = WORLD * T
own = torch.zeros((WORLD * N_GLOB, 1), dtype=torch.int32, device="cuda")
tof = torch.zeros((WORLD * N_GLOB, 1), dtype=torch.int32, device="cuda")
tloc = torch.zeros((WORLD, 1), dtype=torch.int32, device="cuda")
tof_part = torch.zeros((WORLD * 16, 1), dtype=torch.int32, device="cuda")
cnt = torch.zeros((E, 1), dtype=torch.int32, device="cuda"); erb = torch.zeros((E + 1, 1), dtype=torch.int32, device="cuda")
NCHUNK = (WORLD * T * TOPK + 255) // 256; ccnt = torch.zeros((NCHUNK * E, 1), dtype=torch.int32, device="cuda")
gath = torch.full((T_LOC_MAX, 2), -1, dtype=torch.int32, device="cuda")
sti = torch.zeros((PADMAX, 1), dtype=torch.int32, device="cuda"); swt = torch.zeros((PADMAX, 1), dtype=torch.float32, device=dev)
sei = torch.zeros((PADMAX // 32, 1), dtype=torch.int32, device="cuda"); nvi = torch.zeros((2, 1), dtype=torch.int32, device="cuda")
pull_ptr = torch.zeros((T + 1, 1), dtype=torch.int32, device="cuda"); pull_src = torch.zeros((T * WORLD, 2), dtype=torch.int32, device="cuda")
perr = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
pull_stage_pd = torch.full((T * TOPK, 2), -1, dtype=torch.int32, device="cuda")
pull_cnt_pd = torch.zeros((T,), dtype=torch.int32, device="cuda")
dsort_scratch_pd = torch.zeros((256,), dtype=torch.int32, device="cuda")
pd_grid_count = torch.zeros((1,), dtype=torch.int32, device="cuda")
pd_gbar = torch.zeros((1,), dtype=torch.int32, device="cuda")
pd_sort_count = torch.zeros((1,), dtype=torch.int32, device="cuda")
if PF3_REQUESTED:
    for _pf3_arm, _pf3_st in pf3_state.items():
        _pf3_st.update(
            pull_stage=torch.full((T * TOPK, 2), -1, dtype=torch.int32, device="cuda"),
            pull_cnt=torch.zeros((T,), dtype=torch.int32, device="cuda"),
            scratch=torch.zeros((256,), dtype=torch.int32, device="cuda"),
            gbar=torch.zeros((1,), dtype=torch.int32, device="cuda"),
            sort_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
            grid_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
            sc_dst=torch.zeros((MROWS, NG), dtype=torch.float32, device=dev),
            # k0pf4_dsort's per-CTA expert histogram table. LOCAL scratch, fully overwritten
            # every epoch (never accumulated), so it carries no protocol state.
            hcnt=torch.zeros((PF4_MAXE * PF4_MAXCTA,), dtype=torch.int32, device="cuda"),
        )
if PF5_REQUESTED:
    _pf5_clean_u8 = torch.zeros((MROWS, K), dtype=torch.uint8, device=dev)
    pf5_state.update(
        a_dst=_pf5_clean_u8.view(torch.bfloat16).view(MROWS, K // 2),
        a_dst_u8=_pf5_clean_u8,
        sc_stage=torch.zeros((T_LOC_MAX, NG), dtype=torch.float32, device=dev),
        recv_eid=torch.full((T_LOC_MAX, TOPK), -1, dtype=torch.int32, device="cuda"),
        recv_wgt=torch.zeros((T_LOC_MAX, TOPK), dtype=torch.float32, device=dev),
        pushed_count=torch.zeros((WORLD,), dtype=torch.int32, device="cuda"),
        push_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        pull_stage=torch.full((T * TOPK, 2), -1, dtype=torch.int32, device="cuda"),
        pull_cnt=torch.zeros((T,), dtype=torch.int32, device="cuda"),
        grid_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        row_cursor=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        scratch=torch.zeros((256,), dtype=torch.int32, device="cuda"),
        gbar=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        sort_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        sc_dst=torch.zeros((MROWS, NG), dtype=torch.float32, device=dev),
        hcnt=torch.zeros((PF4_MAXE * PF4_MAXCTA,), dtype=torch.int32, device="cuda"),
    )
    # Reuse the established PF3 primitive/epoch gate machinery for this state-compatible arm.
    pf3_state["pf5_ll"] = pf5_state
if PF6_REQUESTED:
    pf6_state.update(
        # Clean ingress destinations and metadata are private because PF6 advances and resets
        # its own full-region epoch independently of every split PF3/PF5 composition.
        a_dst=torch.zeros((MROWS, K), dtype=torch.uint8, device=dev),
        sc_stage=torch.zeros((T_LOC_MAX, NG), dtype=torch.float32, device=dev),
        recv_eid=torch.full((T_LOC_MAX, TOPK), -1, dtype=torch.int32, device="cuda"),
        recv_wgt=torch.zeros((T_LOC_MAX, TOPK), dtype=torch.float32, device=dev),
        pushed_count=torch.zeros((WORLD,), dtype=torch.int32, device="cuda"),
        qpush_done=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        pull_stage=torch.full((TBO16_G * T * TOPK, 2), -1, dtype=torch.int32, device="cuda"),
        pull_cnt=torch.zeros((TBO16_G * T,), dtype=torch.int32, device="cuda"),
        sti=torch.zeros((PADMAX,), dtype=torch.int32, device="cuda"),
        swt=torch.zeros((PADMAX,), dtype=torch.float32, device=dev),
        sei=torch.zeros((PADMAX // 32,), dtype=torch.int32, device="cuda"),
        nvi=torch.zeros((2,), dtype=torch.int32, device="cuda"),
        pull_ptr=torch.zeros((TBO16_G * (T + 1),), dtype=torch.int32, device="cuda"),
        pull_src=torch.full((TBO16_G * T * WORLD, 2), -1, dtype=torch.int32, device="cuda"),
        sc_dst=torch.zeros((MROWS, NG), dtype=torch.float32, device=dev),
        scratch=torch.zeros((256,), dtype=torch.int32, device="cuda"),
        gbar=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        hcnt=torch.zeros((PF4_MAXE * PF4_MAXCTA,), dtype=torch.int32, device="cuda"),
        row_cursor=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        a2_done=torch.zeros((PADMAX // 32,), dtype=torch.int32, device="cuda"),
        part_done=torch.zeros((PADMAX // 32,), dtype=torch.int32, device="cuda"),
        combine_done=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        mega_count=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        # v2 certification: row_remaining[r] = local-slot count of receive row r, rewritten
        # per row every epoch by the streaming unpack (no reset needed).
        row_remaining=torch.zeros((T_LOC_MAX,), dtype=torch.int32, device="cuda"),
    )
    R["pf6"]["state"] = dict(
        independent_from_pf5=True,
        symmetric=(
            "a_ll",
            "rows_done",
            "dest_counter",
            "part",
            "part_ready",
            "retired",
        ),
        local=(
            "clean_ingress",
            "route_plan",
            "a2_done",
            "part_done",
            "combine_done",
            "mega_count",
        ),
        max_blocks=PADMAX // 32,
    )

# Independent frozen-plan oracle buffers. k0pf writes the primary buffers above; the byte oracle
# writes these so no output can alias and accidentally make the equality gate vacuous.
fr_own = torch.zeros_like(own); fr_tof = torch.zeros_like(tof); fr_tloc = torch.zeros_like(tloc)
fr_cnt = torch.zeros_like(cnt); fr_erb = torch.zeros_like(erb); fr_ccnt = torch.zeros_like(ccnt)
fr_gath = torch.full_like(gath, -1)
fr_sti = torch.zeros_like(sti); fr_swt = torch.zeros_like(swt); fr_sei = torch.zeros_like(sei)
fr_nvi = torch.zeros_like(nvi); fr_pull_ptr = torch.zeros_like(pull_ptr)
fr_pull_src = torch.zeros_like(pull_src); fr_perr = torch.zeros_like(perr)
a_dst = torch.zeros((MROWS, K // 2), dtype=torch.bfloat16, device=dev); a_dst_u8 = a_dst.view(torch.uint8).view(MROWS, K)
sc_dst = torch.zeros((MROWS, NG), dtype=torch.float32, device=dev)
sc_dst_pf2 = torch.zeros((MROWS, NG), dtype=torch.float32, device=dev)
sc_stage = torch.zeros((T_LOC_MAX, NG), dtype=torch.float32, device=dev)
stage = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
# PUSH: device-side completion state (NEVER host-reset between replays: monotonic, graph-safe).
emask_disp = torch.zeros((1, 1), dtype=torch.int32, device="cuda")   # expected_dispatch_mask(cur)
emask_comb = torch.zeros((1, 1), dtype=torch.int32, device="cuda")   # expected_combine_mask(cur)
gen_do = torch.zeros((1, 1), dtype=torch.int32, device="cuda")       # generation counter, dispatch-only arm
gen_co = torch.zeros((1, 1), dtype=torch.int32, device="cuda")       # generation counter, combine-only arm
gen_fp = torch.zeros((1, 1), dtype=torch.int32, device="cuda")       # generation counter, full-push arm
pperr  = torch.zeros((1, 1), dtype=torch.int32, device="cuda")       # push fail-closed timeout flag (asserted ==0)
# per-arm flag arrays so replaying one push arm never perturbs another's monotonic counters.
df_do, df_do_p = mori_t((WORLD, 1), "int32"); cf_co, cf_co_p = mori_t((WORLD, 1), "int32")
df_fp, df_fp_p = mori_t((WORLD, 1), "int32"); cf_fp, cf_fp_p = mori_t((WORLD, 1), "int32")
# k0d state is deliberately disjoint from the existing push and n2r experiments.  Every flag is
# symmetric, every count is local, all are zeroed once at setup, and none are reset between replay.
k0ds_b1_flags_p = k0ds_b2_flags_p = None
k0ds_b1_count = k0ds_b2_count = None
k0df_quant_flags_p = k0df_comb_flags_p = k0df_arr_flags_p = None
k0df_quant_count = k0df_arr_count = k0df_don_count = None
k0dsq_b2f_arr_flags_p = k0dsq_b2f_ctrl_flags_p = None
k0dsq_b2f_arr_count = k0dsq_b2f_ctrl_count = None
k0dpd_arr_flags_p = None
k0dpd_arr_count = None
if K0D_REQUESTED:
    k0ds_b1_flags, k0ds_b1_flags_p = mori_t((WORLD, 1), "int32")
    k0ds_b2_flags, k0ds_b2_flags_p = mori_t((WORLD, 1), "int32")
    k0ds_b1_flags.zero_(); k0ds_b2_flags.zero_()
    k0ds_b1_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k0ds_b2_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k0df_quant_flags, k0df_quant_flags_p = mori_t((WORLD, 1), "int32")
    k0df_comb_flags, k0df_comb_flags_p = mori_t((WORLD, 1), "int32")
    k0df_arr_flags, k0df_arr_flags_p = mori_t((WORLD, 1), "int32")
    k0df_quant_flags.zero_(); k0df_comb_flags.zero_(); k0df_arr_flags.zero_()
    k0df_quant_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k0df_arr_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k0df_don_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    # Dedicated state for the split-quant, barrier2-only fold. It must not share the fully folded
    # arm's epochs: both graphs may be captured and replayed in the same process.
    k0dsq_b2f_arr_flags, k0dsq_b2f_arr_flags_p = mori_t((WORLD, 1), "int32")
    k0dsq_b2f_arr_flags.zero_()
    k0dsq_b2f_arr_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    # Capture/warm this control against already-satisfied flags, then zero only its flags after
    # capture. Its cumulative count stays four epochs ahead, making every later replay unsatisfied.
    k0dsq_b2f_ctrl_flags, k0dsq_b2f_ctrl_flags_p = mori_t((WORLD, 1), "int32")
    k0dsq_b2f_ctrl_flags.fill_(1 << 20)
    k0dsq_b2f_ctrl_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    k0dpd_arr_flags, k0dpd_arr_flags_p = mori_t((WORLD, 1), "int32")
    k0dpd_arr_flags.zero_()
    k0dpd_arr_count = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    R["k0d"]["states"] = dict(
        standalone=("b1_flags", "b1_count", "b2_flags", "b2_count"),
        fused=("quant_flags", "quant_count", "combine_done_flags", "arrival_flags", "arr_count", "don_count"),
        split_quant_b2_fold=("arrival_flags", "arr_count"),
        reset_free=True,
    )
# N2R folded-rendezvous state: SYMMETRIC epoch flags (peers increment my flags[their rank] remotely)
# + a LOCAL cumulative grid counter.  Both MONOTONIC — zeroed ONCE at setup, never between replays.
n2r_flags, n2r_flags_p = mori_t((WORLD, 1), "int32")
n2r_flags.zero_()
n2r_grid = torch.zeros((1,), dtype=torch.int32, device="cuda")
# k0pf2 folded push completion: one monotonic flag per source rank in every PE and a local
# cumulative completed-CTA counter. Never reset between graph replays.
pf2_flags, pf2_flags_p = mori_t((WORLD, 1), "int32")
pf2_flags.zero_()
pf2_grid = torch.zeros((1,), dtype=torch.int32, device="cuda")
# Dedicated folded-combine state for the fully folded decode composition.
pf2_n2r_flags, pf2_n2r_flags_p = mori_t((WORLD, 1), "int32")
pf2_n2r_flags.zero_()
pf2_n2r_grid = torch.zeros((1,), dtype=torch.int32, device="cuda")
cand_out = torch.zeros((TBO16_G * T, H), dtype=torch.bfloat16, device=dev)
_m15b_stage = (torch.zeros((WORLD * MAXTOK * H,), dtype=torch.bfloat16, device=dev)
               if os.environ.get("K0_M15B", "0") == "1" else None)


def _tbo16_half_write():
    """Half the NEXT mps launch writes (mode 16). Epochs are 1-BASED
    (k0p6_epoch = mega_count + 1): launch N runs epoch N and its deferred M8
    writes out(N-1) at parity (N-1)&1 == _tbo16_launches&1 counted before
    the launch. (Patch-4 fix of the 0-based off-by-one.)"""
    return _tbo16_launches & 1

def _tbo16_half_latest():
    """Half holding the most recently completed out (valid after >=2
    launches): after N launches the newest write was launch N's, at
    parity (N-1)&1 == (_tbo16_launches + 1)&1."""
    return (_tbo16_launches + 1) & 1

def _poison_out(arm=None):
    """exp_32: fill the shared candidate output with a recognisable bf16 NaN.
    Called ONLY from untimed regions (eager gate, soak, post-timing verify)."""
    if MOK_POISON_OUT:
        if MPS_MODE16 and arm == "mps_mega":
            if _tbo16_launches >= 1:   # mode 16 launch 0 writes nothing
                _h = _tbo16_half_write()
                cand_out.view(torch.int16)[_h * T:(_h + 1) * T].fill_(MOK_POISON_I16)
        elif MPS_MODE16:
            # every other arm writes rows [0,T) of the doubled buffer
            cand_out.view(torch.int16)[0:T].fill_(MOK_POISON_I16)
        else:
            cand_out.view(torch.int16).fill_(MOK_POISON_I16)
    else:
        cand_out.zero_()


def _poison_count(arm=None):
    """Elements still holding the poison pattern. Zero in a correct run."""
    if not MOK_POISON_OUT:
        return 0
    if MPS_MODE16 and arm == "mps_mega":
        if _tbo16_launches < 2:
            return 0
        _h = _tbo16_half_latest()
        return int((cand_out.view(torch.int16)[_h * T:(_h + 1) * T] == MOK_POISON_I16).sum().item())
    if MPS_MODE16:
        return int((cand_out.view(torch.int16)[0:T] == MOK_POISON_I16).sum().item())
    return int((cand_out.view(torch.int16) == MOK_POISON_I16).sum().item())


def _poison_rows(limit=16, arm=None):
    if MPS_MODE16 and arm == "mps_mega":
        if _tbo16_launches < 2:
            return []
        _h = _tbo16_half_latest()
        _bad = (cand_out.view(torch.int16)[_h * T:(_h + 1) * T] == MOK_POISON_I16).any(dim=1)
        return torch.nonzero(_bad, as_tuple=False).flatten()[:limit].cpu().tolist()
    if MPS_MODE16:
        _bad = (cand_out.view(torch.int16)[0:T] == MOK_POISON_I16).any(dim=1)
        return torch.nonzero(_bad, as_tuple=False).flatten()[:limit].cpu().tolist()
    _bad = (cand_out.view(torch.int16) == MOK_POISON_I16).any(dim=1)
    return torch.nonzero(_bad, as_tuple=False).flatten()[:limit].cpu().tolist()


def _poison_report(tag, name):
    """Print and return the survivor count. Nonzero means the arm left a row of
    `out` unwritten this epoch -- the exact bug class a stale `out` hides."""
    _n = _poison_count(name)
    if _n:
        print(f"[POISON] {tag} arm={name} rank={rank} survivors={_n} "
              f"first_rows={_poison_rows(arm=name)}", flush=True)
    elif rank == 0:
        print(f"[POISON] {tag} arm={name} survivors=0", flush=True)
    return _n

base_out = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
SPIN = int(os.environ.get("K0_SPIN_LIMIT", "2000000"))   # bounded, fail-closed; low enough that a broken acquire fails FAST            # bounded, fail-closed (shared node)
hbarrier()

# ===================== region corpus (byte-exact restore) =====================
def raw_from(corpus, idx, name, dt):
    return np.fromfile(os.path.join(corpus, f"rank-{rank}", f"{idx}-{name}.raw.bin"), dtype=dt)
def raw(idx, name, dt):
    return raw_from(REGION_CORPUS, idx, name, dt)
# The live router outputs ARE the symmetric route buffers.  Both production and candidate consume
# these exact tensor objects, so a changed route cannot be hidden behind an out-of-graph private
# staging copy.  Server integration must allocate its fixed graph router-output buffers from the
# MoRI symmetric heap, which is a one-time allocation rather than per-call route construction.
topk_ids = my_ids.reshape(T, TOPK)
topk_wgt = my_wgt.reshape(T, TOPK)
if K0_INPUT_MODE == "mok_synthetic":
    if K0_SYNTH_ROUTE:
        raise ValueError("K0_SYNTH_ROUTE cannot be combined with K0_INPUT_MODE=mok_synthetic")
    if (world, TOPK, E, E_GLOBAL, H, INTER) != (8, 8, 32, 256, 7168, 2048):
        raise ValueError(
            "MoK synthetic PF4H inputs require WORLD=8, TOPK=8, E=32, "
            "E_GLOBAL=256, H=7168, INTER=2048"
        )
    _mok_config = MoKSyntheticConfig(
        world_size=world,
        tokens_per_rank=T,
        hidden_dim=H,
        intermediate_dim=INTER,
        num_experts=E_GLOBAL,
        topk=TOPK,
        seed_base=int(os.environ.get("K0_MOK_SEED_BASE", "1234")),
        weight_chunk_elements=int(
            os.environ.get("K0_MOK_WEIGHT_CHUNK_ELEMENTS", str(32 * 1024 * 1024))
        ),
        route_hist_path=os.environ.get("K0_MOK_ROUTE_HIST", "").strip() or None,
    )
    _mok_inputs = generate_mok_synthetic_inputs(rank, dev, _mok_config)
    hidden = _mok_inputs["hidden"]
    topk_ids.copy_(_mok_inputs["topk_experts"])
    topk_wgt.copy_(_mok_inputs["router_weights"])

    def _weight_per_128x128_quant(weight):
        _e, _n, _k = weight.shape
        _blocks = (
            weight.view(_e, _n // 128, 128, _k // 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(_e, -1, 128 * 128)
        )
        _quantized, _scales = aiter.pertoken_quant(
            _blocks, quant_dtype=dtypes.fp8
        )
        _quantized = (
            _quantized.view(_e, _n // 128, _k // 128, 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(_e, _n, _k)
        )
        return _quantized, _scales.view(_e, _n // 128, _k // 128)

    _w1_unshuffled, fc1_scale = _weight_per_128x128_quant(
        _mok_inputs["w13_bf16"]
    )
    _w2_unshuffled, fc2_scale = _weight_per_128x128_quant(
        _mok_inputs["w2_bf16"]
    )
    w1_b = shuffle_weight(_w1_unshuffled)
    w2_b = shuffle_weight(_w2_unshuffled)
    w13 = w1_b.view(torch.uint8).view(torch.bfloat16).reshape(
        E * 2 * INTER, H // 2
    )
    w2c = w2_b.view(torch.uint8).view(torch.bfloat16).reshape(
        E * H, INTER // 2
    )
    if K0_MOK_REP_EXPERTS:
        # ---- M18 static hot-expert replication (mps_mega arm only) ----
        # Pick the replica set: explicit csv, or the route hist's top-K.
        if _m18_csv:
            _m18_rep_ids = [
                int(x) for x in K0_MOK_REP_EXPERTS.split(",") if x.strip()
            ]
        else:
            from synthetic_inputs import _load_route_hist as _m18_load_hist
            if not _mok_config.route_hist_path:
                raise ValueError(
                    "K0_MOK_REP_EXPERTS=<count> requires K0_MOK_ROUTE_HIST"
                )
            _m18_probs, _m18_hist_meta = _m18_load_hist(
                _mok_config.route_hist_path, E_GLOBAL
            )
            _m18_rep_ids = [
                int(i)
                for i in torch.argsort(_m18_probs, descending=True)[:_M18_NREP]
            ]
        if len(_m18_rep_ids) != len(set(_m18_rep_ids)) or any(
            not 0 <= g < E_GLOBAL for g in _m18_rep_ids
        ):
            raise ValueError(f"invalid M18 replica set {_m18_rep_ids}")
        _m18_EL = E + len(_m18_rep_ids)
        # Materialize replica weights.  Foreign owners' logical weights are
        # regenerated bit-identically from their seeds (setup-time only), so
        # every replica slot quantizes to the owner's exact FP8 bytes.
        _m18_owner_cache = {}
        _m18_s13, _m18_s2 = [], []
        for _gid in _m18_rep_ids:
            _own, _li = _gid // E, _gid % E
            if _own == rank:
                _m18_s13.append(_mok_inputs["w13_bf16"][_li])
                _m18_s2.append(_mok_inputs["w2_bf16"][_li])
            else:
                if _own not in _m18_owner_cache:
                    _oin = generate_mok_synthetic_inputs(_own, dev, _mok_config)
                    _m18_owner_cache[_own] = (_oin["w13_bf16"], _oin["w2_bf16"])
                    del _oin
                _m18_s13.append(_m18_owner_cache[_own][0][_li])
                _m18_s2.append(_m18_owner_cache[_own][1][_li])
        _m18_w13_bf16 = torch.cat(
            [_mok_inputs["w13_bf16"], torch.stack(_m18_s13)], dim=0
        )
        _m18_w2_bf16 = torch.cat(
            [_mok_inputs["w2_bf16"], torch.stack(_m18_s2)], dim=0
        )
        del _m18_owner_cache, _m18_s13, _m18_s2
        _m18_w1_unshuf, _m18_fc1 = _weight_per_128x128_quant(_m18_w13_bf16)
        _m18_w2_unshuf, _m18_fc2 = _weight_per_128x128_quant(_m18_w2_bf16)
        _m18_w13 = (
            shuffle_weight(_m18_w1_unshuf)
            .view(torch.uint8)
            .view(torch.bfloat16)
            .reshape(_m18_EL * 2 * INTER, H // 2)
        )
        _m18_w2c = (
            shuffle_weight(_m18_w2_unshuf)
            .view(torch.uint8)
            .view(torch.bfloat16)
            .reshape(_m18_EL * H, INTER // 2)
        )
        del _m18_w1_unshuf, _m18_w2_unshuf, _m18_w13_bf16, _m18_w2_bf16
        torch.cuda.empty_cache()
        # Replication table: header + lut(int8[256]) + bitmap(u32[8]); the
        # owner keeps its owned slot, every other rank points at E + idx.
        _m18_lut = np.full((256,), -1, dtype=np.int8)
        _m18_lut[rank * E : (rank + 1) * E] = np.arange(E, dtype=np.int8)
        _m18_mask_py = [0] * 8
        for _idx, _gid in enumerate(_m18_rep_ids):
            _m18_mask_py[_gid >> 5] |= 1 << (_gid & 31)
            if _gid // E != rank:
                _m18_lut[_gid] = E + _idx
        _m18_bytes = (
            np.array(
                [0x4D313852, len(_m18_rep_ids), _m18_EL,
                 K0_MOK_REP_THRESHOLD],
                dtype=np.uint32,
            ).tobytes()
            + _m18_lut.tobytes()
            + np.array(_m18_mask_py, dtype=np.uint32).tobytes()
        )
        _m18_rep_buf = torch.from_numpy(
            np.frombuffer(_m18_bytes, dtype=np.uint8).copy()
        ).cuda()
        _m19_rep_dec = None
        if K0_MOK_REP_THRESHOLD:
            # u32[2 parity][world][8]; symmetric because every source
            # peer-writes its decision words each epoch.
            _m19_rep_dec, _m19_rep_dec_p = mori_t((2 * WORLD * 8, 1), "int32")
            _m19_rep_dec.zero_()
        _m20 = None
        if K0_MOK_M20:
            _m20 = {}
            _nrep = len(_m18_rep_ids)
            # Per-expert byte sizes of the SHUFFLED fp8/scale tensors.
            _w13_es = w13.numel() * w13.element_size() // E
            _s13_es = fc1_scale.numel() * fc1_scale.element_size() // E
            _w2_es = w2c.numel() * w2c.element_size() // E
            _s2_es = fc2_scale.numel() * fc2_scale.element_size() // E
            _es = [_w13_es, _s13_es, _w2_es, _s2_es]
            _own = [w13, fc1_scale, w2c, fc2_scale]
            _stage_stride = sum(_es)
            # PUSH design: the pool is SYMMETRIC (owners remote-write
            # peers' buffers); sources are the owner's local slices, so
            # no export staging exists.  Uniform carve order preserved.
            _m20["pool"] = []
            _m20["pool_p"] = []
            # Cache mode (duty 0) needs ONE persistent buffer; the extra
            # buffers existed only for the (measured-infeasible) per-step
            # streaming pipeline and overflowed the mori heap at 3x672MB.
            _m20_nbufs = 3 if int(os.environ.get("K0_MOK_M20_DUTY", "0")) else 1
            for _b in range(_m20_nbufs):
                _bufs, _ps = [], []
                for _sz in _es:
                    _t_sym, _p_sym = mori_t((_nrep * _sz, 1), "uint8")
                    _t_sym.zero_()
                    _bufs.append(_t_sym.view(torch.uint8).reshape(-1))
                    _ps.append(_p_sym)
                _m20["pool"].append(_bufs)
                _m20["pool_p"].append(_ps)
            # Pre-warm buffer 0 with the set's owner bytes (bit-exact
            # regeneration, the established M18 path).
            _oc = {}
            for _idx, _gid in enumerate(_m18_rep_ids):
                _ow, _li = _gid // E, _gid % E
                if _ow == rank:
                    _srcs = _own
                else:
                    if _ow not in _oc:
                        _oi = generate_mok_synthetic_inputs(_ow, dev, _mok_config)
                        _q13, _f1 = _weight_per_128x128_quant(_oi["w13_bf16"])
                        _q2, _f2 = _weight_per_128x128_quant(_oi["w2_bf16"])
                        _oc[_ow] = [
                            shuffle_weight(_q13), _f1, shuffle_weight(_q2), _f2
                        ]
                        del _oi
                    _srcs = _oc[_ow]
                for _k, (_t, _sz) in enumerate(zip(_srcs, _es)):
                    _flat = _t.contiguous().view(torch.uint8).reshape(-1)
                    _m20["pool"][0][_k][_idx * _sz : (_idx + 1) * _sz].copy_(
                        _flat[_li * _sz : (_li + 1) * _sz]
                    )
            del _oc
            torch.cuda.empty_cache()
            # DECIN: constant for the fixed route.  bit(e) = replica
            # slot exists AND own routed-slot count >= theta.
            _cnt = torch.bincount(
                topk_ids.detach().reshape(-1).to(torch.int64),
                minlength=E_GLOBAL,
            ).cpu()
            _bits = [0] * 8
            for _idx, _gid in enumerate(_m18_rep_ids):
                if _gid // E != rank and int(_cnt[_gid]) >= K0_MOK_REP_THRESHOLD:
                    _bits[_gid >> 5] |= 1 << (_gid & 31)
            _m20["decin"] = torch.from_numpy(
                np.array(_bits, dtype=np.uint32).view(np.uint8).copy()
            ).cuda()
            print(
                f"[M20] rank{rank} nrep={_nrep} PUSH "
                f"pool3x={3*_nrep*_stage_stride>>20}MiB decin_bits="
                f"{sum(bin(b).count('1') for b in _bits)}",
                flush=True,
            )
            # Prefetch duty table: absolute peer VAs computed from the
            # attested heap-descriptor words (identical allocation
            # order => the staging offset is rank-invariant).  Duty =
            # the same set into pool buffer 2 (the harness ring is one
            # layer; the read buffer 0 stays host-pre-warmed).
            _hw = _pf6_mps_heap_words
            _heap_base = int(_hw[0])
            _trip = []  # push duty
            # CACHE mode (default): weights are constant across steps, so
            # steady-state duty is ZERO; the engine exists for rate-limited
            # refresh on set changes (K0_MOK_M20_DUTY experts/invocation).
            _duty = int(os.environ.get("K0_MOK_M20_DUTY", "0"))
            for _idx, _gid in enumerate(_m18_rep_ids):
                if _idx >= _duty or _gid // E != rank:
                    continue  # push only what THIS rank owns
                _li = _gid % E
                for _k, (_t, _sz) in enumerate(zip(_own, _es)):
                    _src = _t.data_ptr() + _li * _sz  # local read
                    _dst_off = (
                        _m20["pool_p"][-1][_k] - _heap_base + _idx * _sz
                    )
                    for _pe in range(WORLD):
                        if _pe == rank:
                            continue
                        _trip.append(
                            (_src, int(_hw[1 + _pe]) + _dst_off, _sz)
                        )
            _tab = np.zeros(2 + 3 * len(_trip), dtype=np.uint64)
            _tab[0] = np.uint64(0x4D323050 | (len(_trip) << 32))
            for _i, (_a, _b, _c) in enumerate(_trip):
                _tab[2 + 3 * _i + 0] = np.uint64(_a)
                _tab[2 + 3 * _i + 1] = np.uint64(_b)
                _tab[2 + 3 * _i + 2] = np.uint64(_c)
            _m20["pftab"] = torch.from_numpy(
                _tab.view(np.uint8).copy()
            ).cuda()
        R["m18_replication"] = {
            "theta": K0_MOK_REP_THRESHOLD,
            "rep_ids": _m18_rep_ids,
            "nrep": len(_m18_rep_ids),
            "EL": _m18_EL,
            "owners": sorted({g // E for g in _m18_rep_ids}),
            "source": "csv" if _m18_csv else "route_hist_topk",
            "lut_sha256": hashlib.sha256(_m18_bytes).hexdigest(),
            "w13_ext_shape": list(_m18_w13.shape),
            "w2c_ext_shape": list(_m18_w2c.shape),
        }
        print(
            f"[M18] rank{rank} replicating {len(_m18_rep_ids)} experts "
            f"{_m18_rep_ids} EL={_m18_EL} lut_self_slots="
            f"{int((_m18_lut >= 0).sum())}",
            flush=True,
        )
    del _w1_unshuffled, _w2_unshuffled
    del _mok_inputs["w13_bf16"], _mok_inputs["w2_bf16"]
    ref = None
    R["synthetic_inputs"] = {
        **_mok_inputs["metadata"],
        "mode": "mok_synthetic",
        "timed_generation": False,
        "reference": "pending same-run production API",
    }
    R["synthetic_inputs"]["weights"].update(
        {
            "runtime_w13_shape": list(w1_b.shape),
            "runtime_w2_shape": list(w2_b.shape),
            "fc1_scale_shape": list(fc1_scale.shape),
            "fc2_scale_shape": list(fc2_scale.shape),
            "runtime_dtype": str(w1_b.dtype),
            "layout": "AITER shuffle_weight production layout",
        }
    )
    R["synthetic_route"] = dict(enabled=False)
else:
    hidden = torch.from_numpy(raw("12", "hidden_states", np.uint16)).cuda().view(torch.bfloat16).reshape(T, H)
    topk_ids.copy_(torch.from_numpy(raw("13", "topk_ids", np.int32)).reshape(T, TOPK))
    topk_wgt.copy_(torch.from_numpy(raw("14", "topk_weights", np.float32)).reshape(T, TOPK))
    ref = raw("15", "moe_out", np.uint16).reshape(T, H)
    if K0_SYNTH_ROUTE:
        if (world, T, TOPK, E) != (8, 64, 8, 32):
            raise ValueError(
                "synthetic routes are decode-only and require WORLD=8,T=64,TOPK=8,E=32; "
                f"got {(world, T, TOPK, E)}"
            )
        _synth_ids, _synth_weights, _synth_stats = generate_synthetic_route(
            K0_SYNTH_ROUTE, seed=K0_SYNTH_SEED, world=world, tokens_per_rank=T,
            topk=TOPK, experts_per_rank=E,
        )
        topk_ids.copy_(torch.from_numpy(_synth_ids[rank]))
        if K0_SYNTH_WEIGHT_MODE == "generated":
            topk_wgt.copy_(torch.from_numpy(_synth_weights[rank]))
        _local_weight_sums = topk_wgt.sum(dim=1)
        _synth_stats.update(
            weight_mode=K0_SYNTH_WEIGHT_MODE,
            local_weight_sum_min=float(_local_weight_sums.min().item()),
            local_weight_sum_max=float(_local_weight_sums.max().item()),
            hidden_source="captured",
            model_weights_source="captured",
            timed_generation=False,
        )
        R["synthetic_route"] = _synth_stats
    else:
        R["synthetic_route"] = dict(enabled=False)
    w13 = torch.from_numpy(raw("03", "gate", np.uint8)).cuda().view(torch.bfloat16).reshape(E * 4096, K // 2)
    w2c = torch.from_numpy(raw("04", "down", np.uint8)).cuda().view(torch.bfloat16).reshape(E * K, 2048 // 2)
    fc1_scale = torch.from_numpy(raw("10", "fc1_scale", np.float32)).cuda().reshape(E, 32, NG)
    fc2_scale = torch.from_numpy(raw("11", "fc2_scale", np.float32)).cuda().reshape(E, NG, 16)
    # n2 sees packed FP8 bytes through bf16 containers; AITER sees the same allocations as FP8.
    # Loading a second copy duplicated exactly 1.3125 GiB/rank and changed warm-cache addresses.
    w1_b = w13.view(torch.uint8).view(torch.float8_e4m3fn).reshape(E, 2 * INTER, H)
    w2_b = w2c.view(torch.uint8).view(torch.float8_e4m3fn).reshape(E, H, INTER)
R["weight_storage"] = dict(
    shared_between_production_and_candidate=True,
    w13_w1_same_ptr=bool(w13.data_ptr() == w1_b.data_ptr()),
    w2_w2c_same_ptr=bool(w2c.data_ptr() == w2_b.data_ptr()),
    duplicate_bytes_removed=1409286144,
)
R["k0d"]["route_input_contract"] = dict(
    live_topk_is_symmetric=True,
    ids_same_ptr=bool(topk_ids.data_ptr() == my_ids.data_ptr()),
    weights_same_ptr=bool(topk_wgt.data_ptr() == my_wgt.data_ptr()),
)
expert_mask = torch.ones((E_GLOBAL + 1,), dtype=torch.int32, device="cuda"); expert_mask[-1] = 0
m = torch.zeros((E_GLOBAL,), dtype=torch.int32, device="cuda"); m[rank * E:(rank + 1) * E] = 1
expert_mask[:E_GLOBAL] = m

# The distinct same-shape route corpus is an in-place graph-replay gate, not a second graph capture.
# Retain both input sets and the symmetric route copies so the harness can swap and restore them
# without changing graph addresses or the capture-resident allocation topology.
k0d_route = None
# k0d and pf4h use the same in-place captured-route swap mechanism.  PF4H deliberately
# has no prefill default: the decode default above fails its shape read and blocks timing
# until the caller supplies a distinct, safe prefill corpus through K0_ROUTE_SWAP_CORPUS.
# Do this first collective unconditionally.  A rank-local K0_ARMS disagreement must not let one
# rank open the swap corpus and enter setup collectives while another skips straight to hbarrier.
_ROUTE_SWAP_REQUIRED = K0_BENCHMARK_PROTOCOL == "graph"
R["pf3"]["route_swap_required"] = _ROUTE_SWAP_REQUIRED
_route_request_local = dict(
    k0d=bool(K0D_REQUESTED and _ROUTE_SWAP_REQUIRED),
    pf4h=bool(PF4H_REQUESTED and _ROUTE_SWAP_REQUIRED),
)
_route_requests_by_rank = [None for _ in range(world)]
dist.all_gather_object(_route_requests_by_rank, _route_request_local)
_ROUTE_SWAP_REQUESTED_ANY = any(
    request["k0d"] or request["pf4h"] for request in _route_requests_by_rank
)
_PF4H_REQUESTED_ANY = any(request["pf4h"] for request in _route_requests_by_rank)
if _ROUTE_SWAP_REQUESTED_ANY:
    k0d_route = dict(ready=False, corpus=K0_ROUTE_SWAP_CORPUS)
    try:
        hidden_alt = torch.from_numpy(
            raw_from(K0_ROUTE_SWAP_CORPUS, "12", "hidden_states", np.uint16)
        ).cuda().view(torch.bfloat16).reshape(T, H)
        topk_ids_alt = torch.from_numpy(
            raw_from(K0_ROUTE_SWAP_CORPUS, "13", "topk_ids", np.int32)
        ).cuda().reshape(T, TOPK).contiguous()
        topk_wgt_alt = torch.from_numpy(
            raw_from(K0_ROUTE_SWAP_CORPUS, "14", "topk_weights", np.float32)
        ).cuda().reshape(T, TOPK).contiguous()
        ref_alt = raw_from(K0_ROUTE_SWAP_CORPUS, "15", "moe_out", np.uint16).reshape(T, H)
        k0d_route.update(dict(
            ready=True,
            hidden_orig=hidden.clone(), topk_ids_orig=topk_ids.clone(), topk_wgt_orig=topk_wgt.clone(),
            hidden_alt=hidden_alt, topk_ids_alt=topk_ids_alt, topk_wgt_alt=topk_wgt_alt,
            ref_alt=ref_alt,
        ))
    except Exception as _e_k0d_route:
        k0d_route["error"] = "".join(
            traceback.format_exception_only(type(_e_k0d_route), _e_k0d_route)
        ).strip()
    if K0D_REQUESTED:
        R["k0d"]["route_swap_setup"] = dict(
            ready=bool(k0d_route["ready"]), corpus=K0_ROUTE_SWAP_CORPUS,
            error=k0d_route.get("error"),
        )
    if _PF4H_REQUESTED_ANY:
        # The decisive direction is capture on fewer/equal blocks and replay on a route that
        # needs strictly more.  This is the full destination-rank x local-expert BM32 vector,
        # not merely an aggregate nvi total, so it refuses any over-dispatch component that can
        # be unsafe on the shared node.
        def _prefill_route_metadata(corpus):
            expert_rows = np.zeros((world, E), dtype=np.int64)
            arrival_rows = np.zeros((world,), dtype=np.int64)
            local_weight_bits = [[] for _ in range(E)]
            local_primary = np.full((T, TOPK), -1, dtype=np.int32)
            expected_ids = T * TOPK
            for src_rank in range(world):
                ids = np.fromfile(
                    os.path.join(corpus, f"rank-{src_rank}", "13-topk_ids.raw.bin"),
                    dtype=np.int32,
                )
                if ids.size != expected_ids:
                    raise ValueError(
                        f"{corpus}/rank-{src_rank}/13-topk_ids.raw.bin has {ids.size} IDs; "
                        f"expected {expected_ids} for T={T}, TOPK={TOPK}"
                )
                if np.any(ids < 0) or np.any(ids >= world * E):
                    raise ValueError(f"{corpus}/rank-{src_rank} has an out-of-range expert ID")
                ids = ids.reshape(T, TOPK)
                dests = ids // E
                np.add.at(expert_rows, (dests.reshape(-1), (ids % E).reshape(-1)), 1)
                for dst_rank in range(world):
                    arrival_rows[dst_rank] += int(np.count_nonzero(np.any(dests == dst_rank, axis=1)))
                weights = np.fromfile(
                    os.path.join(corpus, f"rank-{src_rank}", "14-topk_weights.raw.bin"),
                    dtype=np.float32,
                )
                if weights.size != expected_ids:
                    raise ValueError(
                        f"{corpus}/rank-{src_rank}/14-topk_weights.raw.bin has {weights.size} weights; "
                        f"expected {expected_ids} for T={T}, TOPK={TOPK}"
                    )
                weights = weights.reshape(T, TOPK)
                for local_expert in range(E):
                    mask = ids == rank * E + local_expert
                    local_weight_bits[local_expert].extend(weights[mask].view(np.uint32).tolist())
                if src_rank == rank:
                    for token in range(T):
                        seen = set()
                        for slot, dst_rank in enumerate(dests[token]):
                            dst_rank = int(dst_rank)
                            if dst_rank not in seen:
                                local_primary[token, slot] = dst_rank
                                seen.add(dst_rank)
            block_vector = ((expert_rows + 31) // 32).astype(np.int64)
            local_fanout = (local_primary >= 0).sum(axis=1, dtype=np.int64)
            local_pull_ptr = np.empty((T + 1,), dtype=np.int64)
            local_pull_ptr[0] = 0
            np.cumsum(local_fanout, out=local_pull_ptr[1:])
            return dict(
                block_vector=block_vector.tolist(),
                block_totals=[int(x) for x in block_vector.sum(axis=1)],
                arrival_rows=[int(x) for x in arrival_rows],
                local_expert_rows=[int(x) for x in expert_rows[rank]],
                local_weight_bits=[np.sort(np.asarray(x, dtype=np.uint32)) for x in local_weight_bits],
                local_primary=local_primary,
                local_pull_ptr=local_pull_ptr,
            )

        k0d_route["pf4h_metadata_ready"] = False
        if k0d_route["ready"]:
            try:
                metadata_before = _prefill_route_metadata(REGION_CORPUS)
                metadata_after = _prefill_route_metadata(K0_ROUTE_SWAP_CORPUS)
                block_vector_before = metadata_before["block_vector"]
                block_vector_after = metadata_after["block_vector"]
                non_decreasing = all(
                    after >= before
                    for before_rank, after_rank in zip(block_vector_before, block_vector_after)
                    for before, after in zip(before_rank, after_rank)
                )
                has_more = any(
                    after > before
                    for before_rank, after_rank in zip(block_vector_before, block_vector_after)
                    for before, after in zip(before_rank, after_rank)
                )
                k0d_route.update(dict(
                    pf4h_metadata_ready=True,
                    pf4h_metadata_before=metadata_before,
                    pf4h_metadata_after=metadata_after,
                    pf4h_blocks_before=metadata_before["block_totals"],
                    pf4h_blocks_after=metadata_after["block_totals"],
                    pf4h_block_vector_before=block_vector_before,
                    pf4h_block_vector_after=block_vector_after,
                    pf4h_block_vector_axes="destination_rank x local_expert BM32 blocks",
                    pf4h_non_decreasing=non_decreasing,
                    pf4h_has_more_blocks=has_more,
                    pf4h_safe_direction=bool(non_decreasing and has_more),
                ))
            except Exception as _e_pf4h_route:
                k0d_route["pf4h_metadata_error"] = "".join(
                    traceback.format_exception_only(type(_e_pf4h_route), _e_pf4h_route)
                ).strip()
    _route_setup_local = dict(
        ready=bool(k0d_route["ready"]),
        metadata_ready=bool(k0d_route.get("pf4h_metadata_ready", not _PF4H_REQUESTED_ANY)),
        safe_direction=bool(k0d_route.get("pf4h_safe_direction", not _PF4H_REQUESTED_ANY)),
        error=k0d_route.get("error") or k0d_route.get("pf4h_metadata_error"),
    )
    _route_setup_by_rank = [None for _ in range(world)]
    dist.all_gather_object(_route_setup_by_rank, _route_setup_local)
    k0d_route["collective_ready"] = all(x["ready"] for x in _route_setup_by_rank)
    k0d_route["collective_metadata_ready"] = all(
        x["metadata_ready"] for x in _route_setup_by_rank
    )
    k0d_route["collective_safe_direction"] = all(
        x["safe_direction"] for x in _route_setup_by_rank
    )
    k0d_route["setup_by_rank"] = _route_setup_by_rank
    if _PF4H_REQUESTED_ANY:
        R["pf3"]["pf4h_route_swap_setup"] = dict(
            ready=k0d_route["collective_ready"],
            metadata_ready=k0d_route["collective_metadata_ready"],
            safe_direction=k0d_route["collective_safe_direction"],
            requested_by_rank=_route_requests_by_rank,
            corpus=K0_ROUTE_SWAP_CORPUS,
            blocks_before=k0d_route.get("pf4h_blocks_before"),
            blocks_after=k0d_route.get("pf4h_blocks_after"),
            block_vector_before=k0d_route.get("pf4h_block_vector_before"),
            block_vector_after=k0d_route.get("pf4h_block_vector_after"),
            errors=[x["error"] for x in _route_setup_by_rank],
        )
hbarrier()

# ===================== baseline op =====================
cfg = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16, rank=rank, world_size=world,
    hidden_dim=H, scale_dim=NG, scale_type_size=torch.float32.itemsize,
    max_token_type_size=torch.bfloat16.itemsize, max_num_inp_token_per_rank=MAXTOK,
    num_experts_per_rank=E, num_experts_per_token=TOPK,
    warp_num_per_block=16, block_num=80,
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode, gpu_per_node=world, rdma_block_num=0)
op = mori.ops.EpDispatchCombineOp(cfg)
# production arm gets its OWN op at the SHIPPED MAXTOK so the banked MAXTOK=T win is a
# candidate-side change and does not flatter the denominator.
cfg_prod = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16, rank=rank, world_size=world,
    hidden_dim=H, scale_dim=NG, scale_type_size=torch.float32.itemsize,
    max_token_type_size=torch.bfloat16.itemsize, max_num_inp_token_per_rank=MAXTOK_PROD,
    num_experts_per_rank=E, num_experts_per_token=TOPK,
    warp_num_per_block=16, block_num=80,
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode, gpu_per_node=world, rdma_block_num=0)
op_prod = mori.ops.EpDispatchCombineOp(cfg_prod)
QT = QuantType.per_1x128
hbarrier()

# ===================== plan (built once; also rebuilt INSIDE every region body per §4) =====================
def sp(): return torch.cuda.current_stream().cuda_stream
fn_ag.launch((WORLD,), (256,), 0, sp(), all_ids.data_ptr(), all_wgt.data_ptr(), my_ids_p, my_wgt_p, WORLD, T * TOPK, rank)
k0pf_frozen_plan.build_plan_prod(all_ids, all_wgt, fr_own, fr_tof, fr_tloc, fr_cnt, fr_erb, fr_ccnt,
                                 fr_gath, fr_sti, fr_swt, fr_sei, fr_nvi, fr_pull_ptr, fr_pull_src,
                                 fr_perr, WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, sp())
k0pf_plan.build_plan_prod_stream(all_ids, all_wgt, own, tof, tloc, tof_part, cnt, erb, ccnt,
                                 gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
                                 WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, sp())
torch.cuda.synchronize()
T_loc = int(nvi.cpu().numpy().reshape(-1)[1]); R["T_loc"] = T_loc
_padded = int(nvi.cpu().numpy().reshape(-1)[0])
_pull_n = int(pull_ptr[-1].cpu().numpy().reshape(-1)[0])
_plan_err = int(perr.cpu().numpy().reshape(-1)[0])
_fr_plan_err = int(fr_perr.cpu().numpy().reshape(-1)[0])
_plan_pairs = {
    "own": (own.reshape(-1), fr_own.reshape(-1)),
    "tof": (tof.reshape(-1), fr_tof.reshape(-1)),
    "tloc": (tloc.reshape(-1), fr_tloc.reshape(-1)),
    "gath_live": (gath[:T_loc].reshape(-1), fr_gath[:T_loc].reshape(-1)),
    "sti_live": (sti[:_padded].reshape(-1), fr_sti[:_padded].reshape(-1)),
    "swt_live": (swt[:_padded].reshape(-1), fr_swt[:_padded].reshape(-1)),
    "sei_live": (sei[:_padded // 32].reshape(-1), fr_sei[:_padded // 32].reshape(-1)),
    "nvi": (nvi.reshape(-1), fr_nvi.reshape(-1)),
    "pull_ptr": (pull_ptr.reshape(-1), fr_pull_ptr.reshape(-1)),
    "pull_src_live": (pull_src[:_pull_n].reshape(-1), fr_pull_src[:_pull_n].reshape(-1)),
}
_plan_diffs = {k: int((a != b).sum().item()) for k, (a, b) in _plan_pairs.items()}
_plan_local_ok = bool(_plan_err == 0 and _fr_plan_err == 0 and all(v == 0 for v in _plan_diffs.values()))
_plan_ok_t = torch.tensor(int(_plan_local_ok), dtype=torch.int32, device="cuda")
dist.all_reduce(_plan_ok_t, op=dist.ReduceOp.MIN)
_plan_ok = bool(_plan_ok_t.item())
R["k0pf_plan_equivalence"] = dict(
    pass_all_ranks=_plan_ok, local_pass=_plan_local_ok, diffs=_plan_diffs,
    k0pf_perr=_plan_err, frozen_perr=_fr_plan_err, padded=_padded, pull_n=_pull_n,
)
if rank == 0:
    print(f"[K0PF GATE] plan_equivalence pass={_plan_ok} diffs={_plan_diffs} "
          f"perr={_plan_err}/{_fr_plan_err} padded={_padded} T_loc={T_loc}", flush=True)
hbarrier()
if not _plan_ok:
    raise RuntimeError("k0pf plan-equivalence gate failed; movement kernels were not launched")

# k0pf2 uses the same output tensors and must match the frozen oracle byte-for-byte before its
# peer stores are allowed to run. This also leaves the primary tensors in a valid k0pf2-built
# state for the primitive push gate below.
if _HAS_PF2_PLAN:
    k0pf2_plan.build_plan_prod_stream(
        all_ids, all_wgt, own, tof, tloc, tof_part, cnt, erb, ccnt,
        gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
        WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, sp(),
    )
    torch.cuda.synchronize()
    _pf2_padded = int(nvi.cpu().numpy().reshape(-1)[0])
    _pf2_tloc = int(nvi.cpu().numpy().reshape(-1)[1])
    _pf2_pull_n = int(pull_ptr[-1].cpu().numpy().reshape(-1)[0])
    _pf2_err = int(perr.cpu().numpy().reshape(-1)[0])
    _pf2_diffs = {
        k: int((a != b).sum().item()) for k, (a, b) in _plan_pairs.items()
    }
    _pf2_local_ok = bool(
        _pf2_err == 0
        and _pf2_padded == _padded
        and _pf2_tloc == T_loc
        and _pf2_pull_n == _pull_n
        and all(v == 0 for v in _pf2_diffs.values())
    )
    _pf2_plan_ok = _all_plan = torch.tensor(
        int(_pf2_local_ok), dtype=torch.int32, device="cuda"
    )
    dist.all_reduce(_all_plan, op=dist.ReduceOp.MIN)
    _pf2_plan_ok = bool(_all_plan.item())
    R["k0pf2_plan_equivalence"] = dict(
        pass_all_ranks=_pf2_plan_ok,
        local_pass=_pf2_local_ok,
        diffs=_pf2_diffs,
        perr=_pf2_err,
        padded=_pf2_padded,
        T_loc=_pf2_tloc,
        pull_n=_pf2_pull_n,
    )
    if rank == 0:
        print(
            f"[K0PF2 GATE] plan_equivalence pass={_pf2_plan_ok} "
            f"diffs={_pf2_diffs} perr={_pf2_err} padded={_pf2_padded} "
            f"T_loc={_pf2_tloc}",
            flush=True,
        )
    hbarrier()
    if not _pf2_plan_ok:
        raise RuntimeError("k0pf2 plan-equivalence gate failed; push was not launched")
else:
    R["k0pf2_plan_equivalence"] = dict(
        pass_all_ranks=False, import_error=_PF2_PLAN_ERR
    )
sti1 = sti.reshape(-1); swt1 = swt.reshape(-1); sei1 = sei.reshape(-1); nvi1 = nvi.reshape(-1)
INCL_PLAN = int(os.environ.get("K0_INCL_PLAN", "1"))

# ===================== shared region fragments =====================
def _plan(stream):
    if INCL_PLAN:
        fn_ag.launch((WORLD,), (256,), 0, stream, all_ids.data_ptr(), all_wgt.data_ptr(), my_ids_p, my_wgt_p, WORLD, T * TOPK, rank)
        k0pf_frozen_plan.build_plan_prod(all_ids, all_wgt, own, tof, tloc, cnt, erb, ccnt,
                                         gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
                                         WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, stream)
def _plan_pf(stream):
    if INCL_PLAN:
        fn_ag.launch((WORLD,), (256,), 0, stream, all_ids.data_ptr(), all_wgt.data_ptr(), my_ids_p, my_wgt_p, WORLD, T * TOPK, rank)
        k0pf_plan.build_plan_prod_stream(all_ids, all_wgt, own, tof, tloc, tof_part, cnt, erb, ccnt,
                                         gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
                                         WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, stream)
def _plan_pf2(stream):
    if INCL_PLAN:
        fn_ag.launch((WORLD,), (256,), 0, stream, all_ids.data_ptr(), all_wgt.data_ptr(), my_ids_p, my_wgt_p, WORLD, T * TOPK, rank)
        k0pf2_plan.build_plan_prod_stream(
            all_ids, all_wgt, own, tof, tloc, tof_part, cnt, erb, ccnt,
            gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
            WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, stream,
        )
def _quant(stream):
    fn_quant.launch((T * NG,), (128,), 0, stream, hidden.data_ptr(), a_src_p, sc_src_p, T, K)
def _quant_pf(stream):
    fn_quant_pf.launch((T,), (896,), 0, stream, hidden.data_ptr(), a_src_p, sc_src_p, T, K)
def _gather_pull(stream):                       # frozen dispatch: barrier1 + gather_rows_fp8
    ms.shmem_barrier_on_stream(stream)
    fn_gather.launch((T_LOC_MAX,), (GBLK,), 0, stream, a_dst_u8.data_ptr(), sc_dst.data_ptr(), a_src_p, sc_src_p,
                     gath.data_ptr(), nvi.data_ptr(), sc_stage.data_ptr(), T, T_LOC_MAX, K, rank)
def _gather_pf(stream):
    ms.shmem_barrier_on_stream(stream)
    fn_gather_pf.launch((2048,), (256,), 0, stream, a_dst_u8.data_ptr(), sc_dst.data_ptr(), a_src_p, sc_src_p,
                        gath.data_ptr(), nvi.data_ptr(), sc_stage.data_ptr(), T, T_LOC_MAX, K, rank)
def _dispatch_pf2(stream, fused=False):
    # b1 is both quant publication and the cross-epoch WAR guard for peers reusing a_dst_pf2.
    ms.shmem_barrier_on_stream(stream)
    if fused:
        fn_push_pf2_fused.launch(
            (PF2_PUSH_GRID,), (256,), 0, stream,
            a_src_p, sc_src_p, a_dst_pf2_p, sc_stage_pf2_p,
            own.data_ptr(), tof.data_ptr(), pf2_grid.data_ptr(), pf2_flags_p,
            T, WORLD, rank,
        )
        fn_sc_transpose_wait_pf2.launch(
            (PF2_TRANSPOSE_GRID,), (256,), 0, stream,
            sc_stage_pf2_p, sc_dst_pf2.data_ptr(), nvi.data_ptr(),
            pf2_flags_p, pf2_grid.data_ptr(), pperr.data_ptr(), SPIN,
            PF2_PUSH_GRID, T_LOC_MAX, rank, WORLD,
        )
    else:
        fn_push_pf2.launch(
            (PF2_PUSH_GRID,), (256,), 0, stream,
            a_src_p, sc_src_p, a_dst_pf2_p, sc_stage_pf2_p,
            own.data_ptr(), tof.data_ptr(), T, WORLD, rank,
        )
        # The standalone form deliberately pays a real fabric rendezvous before any local scale
        # transpose or n2 read. It is the conservative prefill candidate.
        ms.shmem_barrier_on_stream(stream)
        fn_sc_transpose_pf2.launch(
            (PF2_TRANSPOSE_GRID,), (256,), 0, stream,
            sc_stage_pf2_p, sc_dst_pf2.data_ptr(), nvi.data_ptr(), T_LOC_MAX,
        )
def _n1g(stream):
    # Zero part BEFORE n1g (its epilogue is an atomic pk_add accumulate). Use a captured torch op on the
    # CURRENT (capture) stream: tile_plan.zero_partial's kernel launches on a default stream that is NOT
    # captured into the graph, so under graph replay part was never zeroed and n1g ACCUMULATED across
    # replays (candidate arm k -> k*output). part[:T_LOC_MAX] covers n1g's [0,T_loc) output region.
    part[:T_LOC_MAX].zero_()
    k0_n1g.n1g_fused_moe(a_dst, sc_dst, w13, fc1_scale, w2c, fc2_scale, sti1, swt1, sei1, nvi1, part, stream)
def _combine_pull(stream):                      # frozen combine: barrier2 + owner-pull accumulate
    ms.shmem_barrier_on_stream(stream)
    fn_combine.launch((T,), (CBLK,), 0, stream, cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
                      stage.data_ptr(), T, H, rank)
def _combine_pf(stream):
    ms.shmem_barrier_on_stream(stream)
    fn_combine_pf.launch((1024,), (256,), 0, stream, cand_out.data_ptr(), part_p,
                         pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(), T, H, rank)

# ===================== k0d decode fragments (B64 only; all pointer nulls are integer 0) =====================
def _k0d_fused(stream, folded, do_quant=True):
    # One 1024-thread ingress owns route all-gather, the complete plan, and origin quant.  folded=True
    # additionally carries arm-2's combine_done head-wait and quant_done tail publication.
    if folded:
        comb_flags_p, quant_flags_p, quant_count_p = (
            k0df_comb_flags_p, k0df_quant_flags_p, k0df_quant_count.data_ptr()
        )
    else:
        comb_flags_p = quant_flags_p = quant_count_p = 0
    fn_k0d_fused.launch(
        (1,), (1024,), 0, stream,
        all_ids.data_ptr(), all_wgt.data_ptr(), my_ids_p, my_wgt_p,
        own.data_ptr(), tof.data_ptr(), tloc.data_ptr(),
        gath.data_ptr(), sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), perr.data_ptr(),
        hidden.data_ptr(), a_src_p, sc_src_p,
        WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX,
        comb_flags_p, quant_flags_p, quant_count_p, pperr.data_ptr(), SPIN, int(do_quant),
    )

def _k0d_gather_zero(stream, folded):
    # This kernel owns part[:T_loc] clear.  The subsequent n2 invocation must not issue a second zero.
    if folded:
        quant_flags_p, quant_count_p = k0df_quant_flags_p, k0df_quant_count.data_ptr()
    else:
        quant_flags_p = quant_count_p = 0
    fn_k0d_gather_zero.launch(
        (128,), (256,), 0, stream,
        a_dst_u8.data_ptr(), sc_dst.data_ptr(), a_src_p, sc_src_p,
        gath.data_ptr(), nvi.data_ptr(), sc_dst.data_ptr(), T, T_LOC_MAX, K, rank,
        part.data_ptr(), quant_flags_p, quant_count_p, pperr.data_ptr(), SPIN, WORLD,
    )

def _n2_nozero(stream):
    # k0d_gather_zero has already cleared the exact live partial rows.  Keeping this separate from
    # _n2_from prevents a hidden second part.zero_() from slipping back into either k0d arm.
    k0_n2.n2_phase1(a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                    _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
    k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                    sti1, swt1, sei1, nvi1, part, stream)

def _k0d_combine(stream, folded):
    # k0d_combine's arm-2 cumulative-counter protocol has a fixed 16-block epoch denominator.
    # Use that same 16x256 launch for both arms so A/B differs only in the rendezvous topology.
    if folded:
        arr_flags_p, arr_count_p = k0df_arr_flags_p, k0df_arr_count.data_ptr()
        don_flags_p, don_count_p = k0df_comb_flags_p, k0df_don_count.data_ptr()
    else:
        arr_flags_p = arr_count_p = don_flags_p = don_count_p = 0
    fn_k0d_combine.launch(
        (16,), (256,), 0, stream,
        cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(),
        T, H, rank, arr_flags_p, arr_count_p, don_flags_p, don_count_p,
        pperr.data_ptr(), SPIN, WORLD,
    )

def _k0d_combine_b2f(stream, control=False):
    # Barrier2-only fold: enable the combine-head edge-C rendezvous, but deliberately omit the
    # combine_done tail because the split-quant arm retains standalone barrier1 on the next replay.
    if control:
        arr_flags_p, arr_count_p = (
            k0dsq_b2f_ctrl_flags_p, k0dsq_b2f_ctrl_count.data_ptr()
        )
    else:
        arr_flags_p, arr_count_p = (
            k0dsq_b2f_arr_flags_p, k0dsq_b2f_arr_count.data_ptr()
        )
    fn_k0d_combine.launch(
        (16,), (256,), 0, stream,
        cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(),
        T, H, rank, arr_flags_p, arr_count_p, 0, 0,
        pperr.data_ptr(),
        int(os.environ.get("K0_SPIN_LIMIT_CTRL", "200000")) if control else SPIN,
        WORLD,
    )

def k0d_s_body(stream):
    # Arm 1: the two standalone reset-free barriers are the sole cross-rank rendezvous kernels.
    _k0d_fused(stream, folded=False)
    fn_k0d_barrier.launch((1,), (64,), 0, stream, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                           pperr.data_ptr(), SPIN, rank, WORLD)
    _k0d_gather_zero(stream, folded=False)
    _n2_nozero(stream)
    fn_k0d_barrier.launch((1,), (64,), 0, stream, k0ds_b2_flags_p, k0ds_b2_count.data_ptr(),
                           pperr.data_ptr(), SPIN, rank, WORLD)
    _k0d_combine(stream, folded=False)

def k0d_sq_body(stream):
    # Diagnostic/recovery arm: keep Kimi's one-CTA route+plan collapse but run the already proven
    # massively parallel quant kernel as a second graph node.  One extra launch is far cheaper
    # than serializing all 3,584 K128 groups onto one CU.
    _k0d_fused(stream, folded=False, do_quant=False)
    _quant(stream)
    fn_k0d_barrier.launch((1,), (64,), 0, stream, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                           pperr.data_ptr(), SPIN, rank, WORLD)
    _k0d_gather_zero(stream, folded=False)
    _n2_nozero(stream)
    fn_k0d_barrier.launch((1,), (64,), 0, stream, k0ds_b2_flags_p, k0ds_b2_count.data_ptr(),
                           pperr.data_ptr(), SPIN, rank, WORLD)
    _k0d_combine(stream, folded=False)

def k0d_sq_b2f_body(stream):
    # One-variable sibling of k0d_sq: identical through ordinary n2 phase2, then replace only the
    # standalone barrier2 launch with k0d_combine's production-style edge-C head rendezvous.
    _k0d_fused(stream, folded=False, do_quant=False)
    _quant(stream)
    fn_k0d_barrier.launch((1,), (64,), 0, stream, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                           pperr.data_ptr(), SPIN, rank, WORLD)
    _k0d_gather_zero(stream, folded=False)
    _n2_nozero(stream)
    _k0d_combine_b2f(stream)

def k0d_f_body(stream):
    # Arm 2: no standalone cross-rank barriers.  Edges A/D, B, and C are in fused/gather/combine.
    _k0d_fused(stream, folded=True)
    _k0d_gather_zero(stream, folded=True)
    _n2_nozero(stream)
    _k0d_combine(stream, folded=True)

def k0d_noquant_downstream_control_body(stream):
    # Dedicated positive control for the epoch gate: rebuild a valid plan but deliberately omit
    # origin quant.  Poisoned a_src/sc_src must flow through k0d gather+zero -> n2 -> k0d combine.
    _plan(stream)
    ms.shmem_barrier_on_stream(stream)
    _k0d_gather_zero(stream, folded=False)
    _n2_nozero(stream)
    ms.shmem_barrier_on_stream(stream)
    _k0d_combine(stream, folded=False)

def k0d_sq_b2f_wait_control_body(stream):
    # This control starts one epoch ahead of every peer flag.  A graph replay must therefore hit
    # k0d_combine's bounded edge-C wait and set pperr bit 8.  If it completes with pperr==0, the
    # candidate's "fold" is vacuous and timing must remain blocked.
    _k0d_combine_b2f(stream, control=True)

def _k0d_push_ingress(stream):
    fn_k0d_push.launch(
        (16,), (256,), 0, stream,
        my_ids_p, my_wgt_p, a_src_p, sc_src_p,
        a_dst_pd_p, sc_stage_pd_p, recv_eid_pd_p, recv_wgt_pd_p,
        dest_counter_pd_p, recv_doorbell_pd_p,
        pull_stage_pd.data_ptr(), pull_cnt_pd.data_ptr(), pd_grid_count.data_ptr(),
        pperr.data_ptr(),
        T, TOPK, E, rank, WORLD, T_LOC_MAX,
    )

def _k0d_dsort_ingress(stream):
    fn_k0d_dsort.launch(
        (16,), (256,), 0, stream,
        recv_eid_pd_p, recv_wgt_pd_p, sc_stage_pd_p,
        dest_counter_pd_p, recv_doorbell_pd_p,
        pull_stage_pd.data_ptr(), pull_cnt_pd.data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), sc_dst.data_ptr(), part.data_ptr(),
        dsort_scratch_pd.data_ptr(), pd_gbar.data_ptr(), pd_sort_count.data_ptr(),
        pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
    )

def _k0d_mega_ingress(stream):
    fn_k0d_mega.launch(
        (16,), (256,), 0, stream,
        my_ids_p, my_wgt_p, a_src_p, sc_src_p,
        a_dst_pd_p, sc_stage_pd_p, recv_eid_pd_p, recv_wgt_pd_p,
        dest_counter_pd_p, recv_doorbell_pd_p,
        pull_stage_pd.data_ptr(), pull_cnt_pd.data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), sc_dst.data_ptr(), part.data_ptr(),
        dsort_scratch_pd.data_ptr(), pd_grid_count.data_ptr(), pd_gbar.data_ptr(),
        pd_sort_count.data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
    )

def _n2_nozero_pd(stream):
    k0_n2.n2_phase1(
        a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
        _n2_bufs["a2q"], _n2_bufs["dq2"], stream,
    )
    k0_n2.n2_phase2(
        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
        sti1, swt1, sei1, nvi1, part, stream,
    )

def _k0d_combine_pd(stream):
    fn_k0d_combine.launch(
        (16,), (256,), 0, stream,
        cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
        stage.data_ptr(), T, H, rank,
        k0dpd_arr_flags_p, k0dpd_arr_count.data_ptr(), 0, 0,
        pperr.data_ptr(), SPIN, WORLD,
    )

def k0d_pd_body(stream):
    _quant(stream)
    _k0d_push_ingress(stream)
    _k0d_dsort_ingress(stream)
    _n2_nozero_pd(stream)
    _k0d_combine_pd(stream)

def k0d_mega_body(stream):
    _quant(stream)
    _k0d_mega_ingress(stream)
    _n2_nozero_pd(stream)
    _k0d_combine_pd(stream)

def _k0d_mega_ts_ingress(stream):
    # DIAGNOSTIC ONLY: identical launch to _k0d_mega_ingress plus the trailing stamp buffer.
    fn_k0d_mega_ts.launch(
        (16,), (256,), 0, stream,
        my_ids_p, my_wgt_p, a_src_p, sc_src_p,
        a_dst_pd_p, sc_stage_pd_p, recv_eid_pd_p, recv_wgt_pd_p,
        dest_counter_pd_p, recv_doorbell_pd_p,
        pull_stage_pd.data_ptr(), pull_cnt_pd.data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), sc_dst.data_ptr(), part.data_ptr(),
        dsort_scratch_pd.data_ptr(), pd_grid_count.data_ptr(), pd_gbar.data_ptr(),
        pd_sort_count.data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
        _mega_ts_buf.data_ptr(),
    )

def k0d_mega_ts_body(stream):
    _quant(stream)
    _k0d_mega_ts_ingress(stream)
    _n2_nozero_pd(stream)
    _k0d_combine_pd(stream)

# ===================== ON-TARGET COMPUTE-SWAP BRIDGE (arm f) =====================
# MORI dispatch (BYTE-IDENTICAL to arm a) -> [moe_sorting + per_1x128 quant] -> n1g exact-FP8 GEMM
# -> MORI combine (BYTE-IDENTICAL to arm a). The ONLY variable vs production is the GEMM kernel
# (n1g replaces fmoe_fp8_blockscale_g1u1). This is the "does the COMPUTE beat production" arm.
# Codex 019f8db1 corrections baked in: (1) sort with GLOBAL E=257 + expert_mask (d_ids are global
# expert ids); (2) stock order = SORT then QUANT; (3) transpose_scale=True group-major scale + device
# num_rows=recv; (4) use the sorter's zeroed moe_buf as n1g's output (n1g needs a zeroed live prefix).
_bt = {}                                          # bridge-tensor retainer (graph-safety: hold refs past replay)
GLOBAL_E = E_GLOBAL + 1                            # 257 = expert_mask.numel()

def _bridge_sort(d_ids, d_wgt, recv):
    # returns sorted_ids[40984], sorted_weights[40984], sorted_expert_ids[1281], num_valid_ids[2], moe_buf[M,H]
    return moe_sorting(d_ids, d_wgt, GLOBAL_E, H, torch.bfloat16, 32, expert_mask, recv, 0,
                       accumulate=True, flat=False)

def _bridge_quant(d_a1, recv, transpose_scale=True):
    # stock per_1x128 activation quant: fp8 E4M3 bytes + group-major fp32 block scale (transpose_scale=True).
    # transpose_scale=False is the bridge-specific POSITIVE CONTROL (token-major scale -> n1g reads wrong layout).
    return _hipquant(d_a1, scale=None, quant_dtype=w1_b.dtype, num_rows=recv, transpose_scale=transpose_scale)

def compute_swap_body(stream, transpose_scale=True):     # (f) compute-only swap into UNCHANGED MORI transport
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]     # IDENTICAL trim to prod_body
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    # n1g replaces fmoe_fp8_blockscale_g1u1; fp8 input passed as its BF16 storage view (zero-copy).
    k0_n1g.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                         sti_f, swt_f, sei_f, nvi_f, moe_buf, stream)
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to prod_body
    _ph["compute_swap"] = res
    _bt.update(dict(sti=sti_f, swt=swt_f, sei=sei_f, nvi=nvi_f, moe_buf=moe_buf, a1=a1_f, a1s=a1s_f,
                    d_a1=d_a1, d_ids=d_ids, d_wgt=d_wgt, recv=recv, res=res))

def bridge_matched_diag(stream, transpose_scale=True):
    # UNTIMED matched-pair diagnostic (Codex Q2 #1/#2): ONE dispatch -> ONE sort+quant -> BOTH the production
    # GEMM (fmoe_fp8_blockscale_g1u1, always correct-scale) AND n1g, on IDENTICAL sort/quant metadata, into
    # SEPARATE zeroed buffers. This is the ONLY rigorous way to isolate n1g-vs-fmoe compute: two separate MORI
    # dispatches can assign different receive-row orderings (cross-rank atomic slot alloc), so cross-dispatch
    # row-i comparison is invalid. transpose_scale gates ONLY n1g's scale (the false-scale positive control);
    # fmoe always gets the correct group-major scale as the aligned reference. Outputs are clone()-snapshotted
    # because op.combine returns a VIEW of the op's persistent output buffer (a later combine overwrites it).
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_fmoe = _bridge_sort(d_ids, d_wgt, recv)      # buf_fmoe = sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                            # correct group-major scale (fmoe ref)
    a1s_n1g = a1s if transpose_scale else _bridge_quant(d_a1, recv, False)[1]   # n1g scale (maybe token-major=WRONG)
    aiter.fmoe_fp8_blockscale_g1u1(buf_fmoe, a1, w1_b, w2_b, sti, swt, sei, nvi, TOPK,
                                   a1s, fc1_scale, fc2_scale, "", fc2_smooth_scale=None,
                                   activation=ActivationType.Silu, fc_scale_blkn=128, fc_scale_blkk=128,
                                   block_size_M=32)
    buf_n1g = torch.zeros_like(buf_fmoe)                                 # n1g needs its own zeroed live prefix
    k0_n1g.n1g_fused_moe(a1.view(torch.bfloat16), a1s_n1g, w13, fc1_scale, w2c, fc2_scale,
                         sti, swt, sei, nvi, buf_n1g, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())                               # num_valid_ids[1] = live token count T
    res_fmoe = op.combine(buf_fmoe, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    res_n1g  = op.combine(buf_n1g,  None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_fmoe=buf_fmoe, buf_n1g=buf_n1g, res_fmoe=res_fmoe, res_n1g=res_n1g, live=live)

# ===================== N2 TWO-PHASE COMPUTE-SWAP ARM (arm n, exp_47) =====================
# IDENTICAL to compute_swap_body except the GEMM is k0_n2.n2_phase1 + k0_n2.n2_phase2
# (W13 -> global A2 handoff -> split-N full-K W2, write-once epilogue) instead of the
# k0_n1g oracle. MORI dispatch/sort/quant/combine BYTE-IDENTICAL to compute_swap, so:
#   n2/production           = THE N2 HEADLINE (does the write-once W2 close the prod gap?)
#   n2/compute_swap(oracle) = the two-phase epilogue redesign ALONE (same-session).
# The A2 handoff buffers are SETUP-ONLY allocations (outside the timed graph; phase 1
# rewrites every live row each replay, phase 2 reads only what phase 1 wrote).
_n2_bufs = {}
if _HAS_N2:
    try:
        # Setup must not call only one half of the stateful MoRI dispatch/combine
        # pair. The previous sizing probe left ``op`` half-open, corrupting the
        # first eager compute_swap_n2 invocation while later graph replay happened
        # to pass. AITER's moe_sorting allocation contract is
        #   topk_ids.numel() + num_experts * block_size - topk
        # and the candidate MoRI op's static receive-token capacity is
        # WORLD * MAXTOK.  Compute the exact sorter capacity without touching op.
        _ROWCAP = WORLD * MAXTOK * TOPK + GLOBAL_E * 32 - TOPK
        _n2_bufs["a2q"] = torch.zeros((_ROWCAP, 1024), dtype=torch.bfloat16, device=dev)
        _n2_bufs["dq2"] = torch.zeros((_ROWCAP, 16), dtype=torch.float32, device=dev)
        R["n2_rowcap"] = _ROWCAP
        if PF6_REQUESTED:
            if _ROWCAP < PADMAX:
                raise RuntimeError(
                    f"PF6 N2 row capacity {_ROWCAP} is smaller than PADMAX {PADMAX}"
                )
            # PF6 phase handoff storage is private: a split N2 arm cannot leave bytes or a
            # graph-stable pointer that the full-region arm can observe.
            pf6_state["a2q"] = torch.zeros_like(_n2_bufs["a2q"])
            pf6_state["dq2"] = torch.zeros_like(_n2_bufs["dq2"])
            R["pf6"]["n2_rowcap"] = _ROWCAP
            # v2 descriptor ABI (exp_55 resource gate): the ~50-pointer kernarg list is one
            # device-memory int64 table built ONCE here; the kernel keeps ten kernargs.
            # Offsets mirror K0P6_D_* in prefill_opt/kernels/k0pf6_mega.hip — keep in sync.
            pf6_state["desc"] = torch.tensor(
                [
                    my_ids_p,
                    my_wgt_p,
                    hidden.data_ptr(),
                    pf6_state["a_ll_p"],
                    pf6_state["dest_counter_p"],
                    pf6_state["rows_done_p"],
                    pf6_state["pushed_count"].data_ptr(),
                    pf6_state["qpush_done"].data_ptr(),
                    pf6_state["a_dst"].data_ptr(),
                    pf6_state["sc_stage"].data_ptr(),
                    pf6_state["recv_eid"].data_ptr(),
                    pf6_state["recv_wgt"].data_ptr(),
                    pf6_state["pull_stage"].data_ptr(),
                    pf6_state["pull_cnt"].data_ptr(),
                    pf6_state["sti"].data_ptr(),
                    pf6_state["swt"].data_ptr(),
                    pf6_state["sei"].data_ptr(),
                    pf6_state["nvi"].data_ptr(),
                    pf6_state["pull_ptr"].data_ptr(),
                    pf6_state["pull_src"].data_ptr(),
                    pf6_state["sc_dst"].data_ptr(),
                    pf6_state["part_p"],
                    pf6_state["scratch"].data_ptr(),
                    pf6_state["gbar"].data_ptr(),
                    pf6_state["hcnt"].data_ptr(),
                    pf6_state["row_ready_p"],
                    pf6_state["row_remaining"].data_ptr(),
                    pf6_state["a2q"].data_ptr(),
                    pf6_state["dq2"].data_ptr(),
                    pf6_state["a2_done"].data_ptr(),
                    pf6_state["part_done"].data_ptr(),
                    pf6_state["retired_p"],
                    cand_out.data_ptr(),
                    pf6_state["combine_done"].data_ptr(),
                    pf6_state["mega_count"].data_ptr(),
                    w13.data_ptr(),
                    fc1_scale.data_ptr(),
                    w2c.data_ptr(),
                    fc2_scale.data_ptr(),
                    rank,
                    WORLD,
                    SPIN,
                    pperr.data_ptr(),
                    pf6_state["maxb"],
                    T,
                    TOPK,
                    E,
                    T_LOC_MAX,
                    PADMAX,
                    # bisection knob: k0pf6_mega returns after phase N when nonzero
                    int(os.environ.get("K0_PF6_DEBUG_PHASE", "0")),
                ],
                dtype=torch.int64,
                device=dev,
            )
            assert pf6_state["desc"].numel() == 50, "descriptor layout drift vs k0pf6_mega.hip"
            # exp_56 Option C descriptor: same 50 slots + CHUNK_READY(50) + MAXTOK(51) = 52.
            # MAXTOK == T for this campaign (segment stride = per-source token bound).
            if PF6C_REQUESTED:
                _pf6c_desc_list = pf6_state["desc"].tolist()  # first 50 slots identical
                # slot 49 is the DEBUG bisection knob; allow a pf6c-specific override.
                _pf6c_desc_list[49] = int(
                    os.environ.get("K0_PF6C_DEBUG_PHASE",
                                   os.environ.get("K0_PF6_DEBUG_PHASE", "0"))
                )
                _pf6c_desc_list = _pf6c_desc_list + [
                    pf6_state["chunk_ready_p"],           # K0P6_D_CHUNK_READY = 50
                    int(T),                              # K0P6_D_MAXTOK      = 51 (stride = T)
                    pf6_state["spin_dbg"].data_ptr(),    # K0P6_D_SPIN_DBG    = 52
                ]
                pf6_state["desc_c"] = torch.tensor(
                    _pf6c_desc_list, dtype=torch.int64, device="cuda"
                )
                assert pf6_state["desc_c"].numel() == 53, "pf6c descriptor drift vs k0pf6c_mega.hip"
            # exp_59: pf6gm descriptor = pf6c's 53 slots (it inherits the chunk-release dispatch
            # and combine unchanged) + TILE_DESC(53) + NUM_TILES(54) = 55 slots.
            if PF6GM_REQUESTED:
                _pf6gm_desc_list = pf6_state["desc"].tolist()
                _pf6gm_desc_list[49] = int(
                    os.environ.get("K0_PF6GM_DEBUG_PHASE",
                                   os.environ.get("K0_PF6_DEBUG_PHASE", "0"))
                )
                _pf6gm_desc_list = _pf6gm_desc_list + [
                    pf6_state["chunk_ready_p"],              # K0P6_D_CHUNK_READY = 50
                    int(T),                                 # K0P6_D_MAXTOK      = 51
                    pf6_state["spin_dbg"].data_ptr(),       # K0P6_D_SPIN_DBG    = 52
                    pf6_state["tile_desc"].data_ptr(),      # K0P6_D_TILE_DESC   = 53
                    pf6_state["num_tiles"].data_ptr(),      # K0P6_D_NUM_TILES   = 54
                ]
                pf6_state["desc_gm"] = torch.tensor(
                    _pf6gm_desc_list, dtype=torch.int64, device="cuda"
                )
                assert pf6_state["desc_gm"].numel() == 55, "pf6gm descriptor drift vs k0pf6gm_mega.hip"
            if PF6MPS_REQUESTED:
                if PF6GM_REQUESTED:
                    _pf6mps_donor_desc = list(_pf6gm_desc_list)
                else:
                    _pf6mps_donor_desc = pf6_state["desc"].tolist() + [
                        pf6_state["chunk_ready_p"],
                        int(T),
                        pf6_state["spin_dbg"].data_ptr(),
                        pf6_state["tile_desc"].data_ptr(),
                        pf6_state["num_tiles"].data_ptr(),
                    ]
                if _m18_rep_buf is not None:
                    # M18: the mps arm reads the EL-expert extended tensors;
                    # every other arm keeps the E-expert originals.
                    _pf6mps_donor_desc[35] = _m18_w13.data_ptr()
                    _pf6mps_donor_desc[36] = _m18_fc1.data_ptr()
                    _pf6mps_donor_desc[37] = _m18_w2c.data_ptr()
                    _pf6mps_donor_desc[38] = _m18_fc2.data_ptr()
                _pf6mps_desc_list = k0_mps_host_abi.append_and_validate(
                    _pf6mps_donor_desc,
                    pf6_state["mps_heap_descriptor"].data_ptr(),
                    pf6_state["mps_queue"].data_ptr(),
                    pf6_state["mps_arrivals"].data_ptr(),
                    pf6_state["mps_pushed"].data_ptr(),
                    pf6_state["mps_claim"].data_ptr(),
                    pf6_state["mps_state"].data_ptr(),
                    pf6_state["mps_slots_p"],
                    PF6MPS_CONFIG["word"],
                )
                if _m15b_stage is not None and _m18_rep_buf is not None:
                    raise ValueError(
                        "K0_M15B and K0_MOK_REP_EXPERTS both claim descriptor slot 63"
                    )
                if _m15b_stage is not None:
                    _pf6mps_desc_list = _pf6mps_desc_list + [_m15b_stage.data_ptr()]
                if _m18_rep_buf is not None:
                    _pf6mps_desc_list = _pf6mps_desc_list + [_m18_rep_buf.data_ptr()]
                if _m18_rep_buf is not None and K0_MOK_REP_THRESHOLD:
                    _pf6mps_desc_list = _pf6mps_desc_list + [_m19_rep_dec_p]
                if _m20 is not None:
                    _pf6mps_desc_list = _pf6mps_desc_list + [
                        _m20["decin"].data_ptr(),
                        _m20["pool_p"][0][0],
                        _m20["pool_p"][0][1],
                        _m20["pool_p"][0][2],
                        _m20["pool_p"][0][3],
                        _m20["pftab"].data_ptr(),
                    ]
                _k0_mps_dbg = os.environ.get("K0_MPS_DEBUG_STOP", "").strip()
                if _k0_mps_dbg:
                    _pf6mps_desc_list[49] = int(_k0_mps_dbg)
                pf6_state["desc_mps"] = torch.tensor(
                    _pf6mps_desc_list, dtype=torch.int64, device="cuda"
                )
                _expected_mps_words = 63
                if _m15b_stage is not None or _m18_rep_buf is not None:
                    _expected_mps_words = 64
                if _m18_rep_buf is not None and K0_MOK_REP_THRESHOLD:
                    _expected_mps_words = 65
                if _m20 is not None:
                    _expected_mps_words = 71
                assert pf6_state["desc_mps"].numel() == _expected_mps_words, (
                    "MPS descriptor drift vs k0pf6gm_device_tile_mps.hip"
                )
                _k0_mps_dbg = os.environ.get("K0_MPS_DEBUG_STOP", "").strip()
                if _k0_mps_dbg:
                    _pf6mps_desc_list[49] = int(_k0_mps_dbg)
        torch.cuda.synchronize()
    except Exception as _e_n2b:
        _HAS_N2 = False; _N2_ERR = "n2 buffer setup: " + "".join(traceback.format_exception_only(type(_e_n2b), _e_n2b)).strip()

def compute_swap_n2_body(stream, transpose_scale=True):
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    k0_n2.n2_phase1(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale,
                    sti_f.reshape(-1), sei_f.reshape(-1), nvi_f.reshape(-1),
                    _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
    k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                    sti_f.reshape(-1), swt_f.reshape(-1), sei_f.reshape(-1), nvi_f.reshape(-1),
                    moe_buf, stream)
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to compute_swap_body
    _ph["compute_swap_n2"] = res
    _bt.update(dict(sti_n=sti_f, swt_n=swt_f, sei_n=sei_f, nvi_n=nvi_f, moe_buf_n=moe_buf, a1_n=a1_f, a1s_n=a1s_f,
                    d_a1_n=d_a1, d_ids_n=d_ids, res_n=res))

def n2_matched_diag(stream):
    # ONE dispatch -> ONE sort+quant -> fmoe (oracle reference) AND n2 into SEPARATE zeroed buffers.
    # The DEFINITIVE compute-isolation gate: on IDENTICAL sort/quant metadata, n2 must equal fmoe on
    # the LIVE prefix (only BF16 atomic-order nondeterminism), the same equality bar n1g meets.
    # Only the n2 buffer is combined: the MoRI op is a stateful 1:1 dispatch/combine pair, while the
    # fmoe comparison is already definitive on the row-aligned inner buffers.
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_fmoe = _bridge_sort(d_ids, d_wgt, recv)             # sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                                   # correct group-major scale
    aiter.fmoe_fp8_blockscale_g1u1(buf_fmoe, a1, w1_b, w2_b, sti, swt, sei, nvi, TOPK,
                                   a1s, fc1_scale, fc2_scale, "", fc2_smooth_scale=None,
                                   activation=ActivationType.Silu, fc_scale_blkn=128, fc_scale_blkk=128,
                                   block_size_M=32)
    buf_n2 = torch.zeros_like(buf_fmoe)
    k0_n2.n2_phase1(a1.view(torch.bfloat16), a1s, w13, fc1_scale,
                    sti.reshape(-1), sei.reshape(-1), nvi.reshape(-1),
                    _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
    k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                    sti.reshape(-1), swt.reshape(-1), sei.reshape(-1), nvi.reshape(-1),
                    buf_n2, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())
    res_n2 = op.combine(buf_n2, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_fmoe=buf_fmoe, buf_n2=buf_n2, res_n2=res_n2, live=live)

# ===================== c32 TAIL-AWARE COMPUTE-SWAP ARM (arm g) =====================
# IDENTICAL to compute_swap_body except the GEMM kernel is k0_n1g_c32 (tail-aware BM32-M16 skip) instead of
# the k0_n1g oracle. MORI dispatch/sort/quant/combine are BYTE-IDENTICAL to compute_swap, so:
#   c32/production           = the c32 HEADLINE (does the tail-aware compute close the ~38us prod gap?)
#   c32/compute_swap(oracle) = the tail-aware skip ALONE (same-session; Codex: no cross-run subtraction).
def compute_swap_c32_body(stream, transpose_scale=True):
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    k0_n1g_c32.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                             sti_f, swt_f, sei_f, nvi_f, moe_buf, stream)         # c32 replaces the n1g oracle
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to compute_swap_body
    _ph["compute_swap_c32"] = res
    _bt.update(dict(sti_c=sti_f, swt_c=swt_f, sei_c=sei_f, nvi_c=nvi_f, moe_buf_c=moe_buf, a1_c=a1_f, a1s_c=a1s_f,
                    d_a1_c=d_a1, d_ids_c=d_ids, d_wgt_c=d_wgt, res_c=res))

def c32_matched_diag(stream):
    # ONE dispatch -> ONE sort+quant -> BOTH oracle n1g AND c32 n1g into SEPARATE zeroed buffers. The DEFINITIVE
    # compute-isolation gate: on IDENTICAL sort/quant metadata, c32 must equal the oracle n1g on the LIVE prefix
    # (c32 skips ONLY all-padding rowtiles whose output is already select/xtok<T masked to 0).
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_oracle = _bridge_sort(d_ids, d_wgt, recv)             # sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                                     # correct group-major scale
    buf_c32 = torch.zeros_like(buf_oracle)
    k0_n1g.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                         sti, swt, sei, nvi, buf_oracle, stream)
    k0_n1g_c32.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                             sti, swt, sei, nvi, buf_c32, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())
    res_oracle = op.combine(buf_oracle, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    res_c32    = op.combine(buf_c32,    None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_oracle=buf_oracle, buf_c32=buf_c32, res_oracle=res_oracle, res_c32=res_c32, live=live)

# ===================== AH W13 ADDRESS-HOIST COMPUTE-SWAP ARM (arm h) =====================
# IDENTICAL to compute_swap_body except the GEMM kernel is k0_n1g_ah (W13 B-weight address hoist)
# instead of the k0_n1g oracle. MORI dispatch/sort/quant/combine BYTE-IDENTICAL to compute_swap, so:
#   ah/production           = THE AH HEADLINE (does the hoisted W13 address stream close the ~38us prod gap?)
#   ah/compute_swap(oracle) = the address hoist ALONE (same-session; Codex >=3us kill vs the oracle).
def compute_swap_ah_body(stream, transpose_scale=True):
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    k0_n1g_ah.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                            sti_f, swt_f, sei_f, nvi_f, moe_buf, stream)          # AH replaces the n1g oracle
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to compute_swap_body
    _ph["compute_swap_ah"] = res

def ah_matched_diag(stream):
    # ONE dispatch -> ONE sort+quant -> BOTH oracle n1g AND ah n1g into SEPARATE zeroed buffers. The DEFINITIVE
    # compute-isolation gate: on IDENTICAL sort/quant metadata, ah must equal the oracle n1g on the LIVE prefix
    # (the W13 address hoist streams the SAME bytes -> bit-identical MFMA inputs; only the addressing differs).
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_oracle = _bridge_sort(d_ids, d_wgt, recv)             # sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                                     # correct group-major scale
    buf_ah = torch.zeros_like(buf_oracle)
    k0_n1g.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                         sti, swt, sei, nvi, buf_oracle, stream)
    k0_n1g_ah.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                            sti, swt, sei, nvi, buf_ah, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())
    res_oracle = op.combine(buf_oracle, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    res_ah     = op.combine(buf_ah,     None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_oracle=buf_oracle, buf_ah=buf_ah, res_oracle=res_oracle, res_ah=res_ah, live=live)

# ===================== ECS EXPERT-MAJOR CO-SCHEDULING COMPUTE-SWAP ARM (arm i) =====================
# IDENTICAL to compute_swap_body except the GEMM kernel is k0_n1g_ecs (task walk = (expert,g), not
# (block,g)) instead of the k0_n1g oracle. MORI dispatch/sort/quant/combine BYTE-IDENTICAL to compute_swap, so:
#   ecs/production           = THE ECS HEADLINE (does removing the 1.667x DRAM weight re-read close the ~38us gap?)
#   ecs/compute_swap(oracle) = the co-scheduling ALONE (same-session; Codex >=3us kill vs the oracle).
def compute_swap_ecs_body(stream, transpose_scale=True):
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    k0_n1g_ecs.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                             sti_f, swt_f, sei_f, nvi_f, moe_buf, stream)         # ECS replaces the n1g oracle
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to compute_swap_body
    _ph["compute_swap_ecs"] = res

def ecs_matched_diag(stream):
    # ONE dispatch -> ONE sort+quant -> BOTH oracle n1g AND ecs n1g into SEPARATE zeroed buffers. The DEFINITIVE
    # compute-isolation gate: ecs reorders WHICH CTA runs each (block,g) but computes the identical per-block GEMM
    # on identical A/weights/scales, so it must equal the oracle on the LIVE prefix (only BF16-combine atomic order
    # differs -> near-zero, like ah).
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_oracle = _bridge_sort(d_ids, d_wgt, recv)             # sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                                     # correct group-major scale
    buf_ecs = torch.zeros_like(buf_oracle)
    k0_n1g.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                         sti, swt, sei, nvi, buf_oracle, stream)
    k0_n1g_ecs.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                             sti, swt, sei, nvi, buf_ecs, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())
    res_oracle = op.combine(buf_oracle, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    res_ecs    = op.combine(buf_ecs,    None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_oracle=buf_oracle, buf_ecs=buf_ecs, res_oracle=res_oracle, res_ecs=res_ecs, live=live)

# ===================== ECS-D2 DECISIVE CO-SCHEDULING COMPUTE-SWAP ARM (arm j) =====================
# The DECISIVE co-scheduling candidate (Codex 019f8db1): pair-leader D=2 on the oracle (block,g) walk, per-block
# e reload (no AGPR drain, oracle-equivalent placement). MORI dispatch/sort/quant/combine BYTE-IDENTICAL, so:
#   ecs_d2/production           = THE DECISIVE HEADLINE (does D=2 weight reuse close the ~38us prod gap?)
#   ecs_d2/compute_swap(oracle) = the co-scheduling ALONE (same-session; Codex >=3us kill).
def compute_swap_ecs_d2_body(stream, transpose_scale=True):
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)        # SORT FIRST (stock order)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, transpose_scale)                      # THEN quant (stock order)
    k0_n1g_ecs_d2.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                                sti_f, swt_f, sei_f, nvi_f, moe_buf, stream)      # ECS-D2 replaces the n1g oracle
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]               # IDENTICAL to compute_swap_body
    _ph["compute_swap_ecs_d2"] = res

def ecs_d2_matched_diag(stream):
    # ONE dispatch -> ONE sort+quant -> BOTH oracle n1g AND ecs_d2 into SEPARATE zeroed buffers. ecs_d2 only
    # reorders/pairs WHICH CTA runs each (block,g) but computes the identical per-block GEMM, so it must equal the
    # oracle on the LIVE prefix (only BF16-combine atomic order differs -> near-zero).
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti, swt, sei, nvi, buf_oracle = _bridge_sort(d_ids, d_wgt, recv)             # sorter's zeroed moe_buf
    a1, a1s = _bridge_quant(d_a1, recv, True)                                     # correct group-major scale
    buf_ecs_d2 = torch.zeros_like(buf_oracle)
    k0_n1g.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                         sti, swt, sei, nvi, buf_oracle, stream)
    k0_n1g_ecs_d2.n1g_fused_moe(a1.view(torch.bfloat16), a1s, w13, fc1_scale, w2c, fc2_scale,
                                sti, swt, sei, nvi, buf_ecs_d2, stream)
    torch.cuda.synchronize()
    live = int(nvi.reshape(-1)[1].item())
    res_oracle = op.combine(buf_oracle, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    res_ecs_d2 = op.combine(buf_ecs_d2, None, topk_ids, BLK, -1, WARP)[0][:T].clone()
    return dict(buf_oracle=buf_oracle, buf_ecs_d2=buf_ecs_d2, res_oracle=res_oracle, res_ecs_d2=res_ecs_d2, live=live)

# ---- PUSH fragments (Phase B kernels; these grids/blocks/args ARE the IMPL_SPEC ABI contract) ----
def _emask(stream):                             # device prologue: own[] -> emask_disp/emask_comb (once per replay, no host work)
    fn_emask.launch((1,), (256,), 0, stream, own.data_ptr(), emask_disp.data_ptr(), emask_comb.data_ptr(), WORLD, T, rank)
def _epoch(stream, gen_p):                      # bump the device generation ONCE per replay (drives slot gen&1 AND the acquire compare)
    fn_epoch.launch((1,), (1,), 0, stream, gen_p)
def _dispatch_push(stream, gen_p, df_p):        # PUSH: replaces barrier1 + gather_pull (gen already bumped by the body)
    fn_dpush.launch((T,), (256,), 0, stream, a_src_p, sc_src_p, a_push_p, sc_push_p,      # ShmemPtrP2p direct stores; per-block system fence
                    own.data_ptr(), tof.data_ptr(), tloc.data_ptr(), gen_p, T, T_LOC_MAX, K, NG, WORLD, rank)
    fn_drel.launch((1,), (WORLD,), 0, stream, df_p, own.data_ptr(), WORLD, T, rank)       # one atomic-add per dest cur pushed to
    fn_dacq.launch((1,), (64,), 0, stream, df_p, emask_disp.data_ptr(), gen_p,            # bounded fail-closed spin + system-acquire fence
                   pperr.data_ptr(), SPIN, WORLD, rank)
    fn_dcopy.launch((T_LOC_MAX,), (256,), 0, stream, a_push_p, sc_push_p,                 # materialize slot gen&1 -> n1g's FIXED a_dst/sc_dst
                    a_dst_u8.data_ptr(), sc_dst.data_ptr(), gen_p, nvi.data_ptr(), tloc.data_ptr(), T_LOC_MAX, K, NG, rank)
def _combine_push(stream, gen_p, cf_p):         # PUSH: replaces barrier2 + combine_pull (gen already bumped by the body)
    fn_cpub.launch((WORLD * T,), (256,), 0, stream, part_p, oslots_p, own.data_ptr(),      # grid over GLOBAL tokens; ShmemPtrP2p direct stores -> owner_slots[slot]; per-block fence
                   tof.data_ptr(), tloc.data_ptr(), gen_p, T, H, WORLD, rank)
    fn_crel.launch((1,), (WORLD,), 0, stream, cf_p, own.data_ptr(), WORLD, T, rank)       # one atomic-add per owner cur produced for
    fn_cacq.launch((1,), (64,), 0, stream, cf_p, emask_comb.data_ptr(), gen_p,            # bounded fail-closed spin + system-acquire fence
                   pperr.data_ptr(), SPIN, WORLD, rank)
    fn_reduce.launch((T,), (256,), 0, stream, cand_out.data_ptr(), oslots_p, pull_ptr.data_ptr(),  # fp32 sum in pull_src order -> bf16 RNE
                     pull_src.data_ptr(), gen_p, T, H, WORLD, rank)

# ===================== the five region bodies =====================
# FAIRNESS FIX (problem 11): production must NOT pay a timed output copy the candidate arms avoid.
# Production reads its combine output directly from the returned tensor (its natural moe_out endpoint);
# candidates' combine kernels already write cand_out in place. _ph stashes production's live output view.
_ph = {}
_FP_ARMS = ("fp_n1g_real", "fp_n1g_alias", "fp_aiter_real", "fp_aiter_alias")   # weight-footprint 2x2 (finiteness-only gate)
_PROD_LIKE = ("production", "compute_swap", "compute_swap_c32", "compute_swap_ah", "compute_swap_ecs", "compute_swap_ecs_d2", "compute_swap_n2") + _FP_ARMS   # arms that read their combine output directly (not cand_out)
def _obuf(name):
    if MPS_MODE16 and name != "mps_mega":
        # every non-mps arm writes rows [0,T) of the doubled cand_out
        if name in _CAND_OUT_ARMS:
            return cand_out[0:T]
    if name == "production": return _ph["prod"]
    if name == "compute_swap": return _ph["compute_swap"]
    if name == "compute_swap_n2": return _ph["compute_swap_n2"]
    if name == "compute_swap_c32": return _ph["compute_swap_c32"]
    if name == "compute_swap_ah": return _ph["compute_swap_ah"]
    if name == "compute_swap_ecs": return _ph["compute_swap_ecs"]
    if name == "compute_swap_ecs_d2": return _ph["compute_swap_ecs_d2"]
    if name in _FP_ARMS: return _ph[name]
    if name == "mps_mega" and MPS_MODE16:
        if _tbo16_launches < 2:
            raise RuntimeError(
                "mode 16 deferred combine: gate requested before any "
                "completed epoch (<2 launches)")
        _h = _tbo16_half_latest()
        return cand_out[_h * T:(_h + 1) * T]
    return cand_out

# arms whose combine output lands in cand_out (vs production's own view)
_CAND_OUT_ARMS = ("frozen_n2", "frozen_n2r", "pf6c_mega", "pf6gm_mega", "mps_mega")

# ===================== WEIGHT-FOOTPRINT DIAGNOSTIC (Codex 019f8db1: real-vs-aliased 2x2) =====================
# THE decisive brick before any co-scheduling build. c32 (MFMA count) + ah (W13 address VALU) are RETIRED
# graph-mode washes; the residual ~38us prod deficit is hypothesized as a 1.667x DRAM weight re-read (§L22).
# 2x2 = {n1g oracle, aiter low-level core fmoe_fp8_blockscale_g1u1} x {real sorted_expert_ids, CACHE1 alias}.
#   CACHE1 alias (sei -> expert 0 for every LIVE block): all tasks read the SAME expert weights => LLC-resident,
#   no DRAM weight stream/re-read (== §L22's CACHE1 = 106.76us vs REPLAY 270us on the isolated STOCK GEMM).
# The 4 arms share BYTE-IDENTICAL MORI dispatch -> bridge moe_sorting -> per_1x128 quant -> [FIXED sei] -> GEMM
# -> MORI combine; the ONLY variables are (kernel) and (sei contents). Paired (real-alias) per-kernel region-wall
# delta = that kernel's weight-DRAM cost; DoD = (real gap) - (alias gap) = the portion UNIQUE to n1g.
# Codex design: precompute FIXED same-shaped sei_real/sei_alias buffers OUTSIDE the timed graph (NO captured
# `where` node); alias only sei<E live blocks (padding sentinel untouched -> task-count + instruction coverage
# preserved for BOTH kernels; sei is route-invariant/stable so a fixed snapshot == the live per-replay sort sei).
_HAS_FP = bool(int(os.environ.get("K0_FOOTPRINT", "0"))); _FP_ERR = None
sei_real_fixed = sei_alias_fixed = None
if _HAS_FP:
    try:
        hbarrier()
        _d_a1, _d_wgt, _d_sc, _d_ids, _recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
        _d_a1, _d_wgt, _d_ids = _d_a1[:M_TRIM], _d_wgt[:M_TRIM], _d_ids[:M_TRIM]
        _s0, _w0, _sei0, _nvi0, _buf0 = _bridge_sort(_d_ids, _d_wgt, _recv)
        torch.cuda.synchronize()
        sei_real_fixed = _sei0.clone().contiguous()                 # real block->expert map (stable across replays)
        _nb = int(_nvi0.reshape(-1)[0].item()) // 32                # live 32-blocks == n1g num_tasks/8 (nvi[0]=padded rows)
        sei_alias_fixed = sei_real_fixed.clone()
        sei_alias_fixed.reshape(-1)[:_nb] = 0                        # CACHE1: every live block -> expert 0 (LLC-resident)
        _seir = sei_real_fixed.reshape(-1)
        R["fp_num_blocks"] = _nb
        R["fp_live_experts"] = sorted(set(int(x) for x in _seir[:_nb].detach().cpu().numpy().tolist()))
        R["fp_sentinel_ok"] = bool(bool((_seir[:_nb] < E).all().item()) and
                                   (True if _nb >= _seir.numel() else bool((_seir[_nb] >= E).item())))
        if rank == 0:
            print(f"[MARK] FOOTPRINT precompute num_blocks={_nb} n_live_experts={len(R['fp_live_experts'])} "
                  f"live_experts={R['fp_live_experts']} sentinel_ok={R['fp_sentinel_ok']}", flush=True)
        hbarrier()
    except Exception as _e_fp:
        _HAS_FP = False; _FP_ERR = "".join(traceback.format_exception_only(type(_e_fp), _e_fp)).strip()
R["has_fp"] = _HAS_FP; R["fp_err"] = _FP_ERR

def _fp_body(stream, kernel, fixed_sei, key):
    # BYTE-IDENTICAL to compute_swap_body EXCEPT (kernel) and (the FIXED sei passed to the GEMM). The live sort
    # still runs (work parity); only its sei OUTPUT is bypassed by the precomputed fixed buffer.
    d_a1, d_wgt, d_scale, d_ids, recv = op.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    sti_f, swt_f, sei_f, nvi_f, moe_buf = _bridge_sort(d_ids, d_wgt, recv)   # sei_f IGNORED (fixed_sei used instead)
    a1_f, a1s_f = _bridge_quant(d_a1, recv, True)
    if kernel == "n1g":
        k0_n1g.n1g_fused_moe(a1_f.view(torch.bfloat16), a1s_f, w13, fc1_scale, w2c, fc2_scale,
                             sti_f, swt_f, fixed_sei, nvi_f, moe_buf, stream)
    else:  # aiter low-level core (== production's inner GEMM, validated by oracle_vs_prod)
        aiter.fmoe_fp8_blockscale_g1u1(moe_buf, a1_f, w1_b, w2_b, sti_f, swt_f, fixed_sei, nvi_f, TOPK,
                                       a1s_f, fc1_scale, fc2_scale, "", fc2_smooth_scale=None,
                                       activation=ActivationType.Silu, fc_scale_blkn=128, fc_scale_blkk=128,
                                       block_size_M=32)
    res = op.combine(moe_buf, None, topk_ids, BLK, -1, WARP)[0][:T]
    _ph[key] = res
    _bt[key] = dict(moe_buf=moe_buf, a1=a1_f, a1s=a1s_f, sti=sti_f, swt=swt_f, nvi=nvi_f, res=res)

def fp_n1g_real_body(stream):    _fp_body(stream, "n1g",   sei_real_fixed,  "fp_n1g_real")
def fp_n1g_alias_body(stream):   _fp_body(stream, "n1g",   sei_alias_fixed, "fp_n1g_alias")
def fp_aiter_real_body(stream):  _fp_body(stream, "aiter", sei_real_fixed,  "fp_aiter_real")
def fp_aiter_alias_body(stream): _fp_body(stream, "aiter", sei_alias_fixed, "fp_aiter_alias")

def prod_body(stream):
    d_a1, d_wgt, d_scale, d_ids, recv = op_prod.dispatch(hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
    d_a1, d_wgt, d_ids = d_a1[:M_TRIM], d_wgt[:M_TRIM], d_ids[:M_TRIM]
    fused = fused_moe(d_a1, w1_b, w2_b, d_wgt, d_ids, expert_mask=expert_mask,
                      activation=ActivationType.Silu, quant_type=QT,
                      w1_scale=fc1_scale, w2_scale=fc2_scale, a1_scale=None,
                      num_local_tokens=recv, doweight_stage1=False, dtype=torch.bfloat16)
    res = op_prod.combine(fused, None, topk_ids, BLK, -1, WARP)[0][:T]
    _ph["prod"] = res   # FAIRNESS FIX: NO timed base_out.copy_(res). Gate reads res directly.

# A synthetic route has no captured ``15-moe_out``. Run the actual production body once, outside
# graph capture and timing, and use its byte output as the numerical oracle for every gate.
if K0_SYNTH_ROUTE or K0_INPUT_MODE == "mok_synthetic":
    hbarrier()
    prod_body(sp())
    torch.cuda.synchronize()
    ref = _ph["prod"].view(torch.uint16).cpu().numpy().copy()
    if K0_INPUT_MODE == "mok_synthetic":
        R["synthetic_inputs"]["reference"] = (
            "same-run production API outside capture/timing"
        )
    else:
        R["synthetic_route"]["reference"] = "same-run production API outside capture/timing"
        R["synthetic_route"]["enabled"] = True
    hbarrier()

def _n2_from(stream, a_in, sc_in):
    # n2 two-phase GEMM on a selected dispatch destination -> part.
    part[:T_LOC_MAX].zero_()
    k0_n2.n2_phase1(a_in, sc_in, w13, fc1_scale, sti1, sei1, nvi1,
                    _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
    k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                    sti1, swt1, sei1, nvi1, part, stream)

def _n2(stream):
    # Frozen/k0pf pull destination.
    _n2_from(stream, a_dst, sc_dst)

def _n2_pf2(stream):
    # k0pf2 direct-push symmetric destination.
    _n2_from(stream, a_dst_pf2, sc_dst_pf2)

def _pf3_qpush(stream, st):
    fn_pf3_qpush.launch(
        (PF3_QPUSH_GRID,), (256,), 0, stream,
        my_ids_p, my_wgt_p, hidden.data_ptr(),
        st["a_dst_p"], st["sc_stage_p"], st["recv_eid_p"], st["recv_wgt_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        st["grid_count"].data_ptr(), pperr.data_ptr(),
        T, TOPK, E, rank, WORLD, T_LOC_MAX,
    )

def _pf3_dsort(stream, st):
    fn_pf3_dsort.launch(
        (PF3_DSORT_GRID,), (256,), 0, stream,
        st["recv_eid_p"], st["recv_wgt_p"], st["sc_stage_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), st["sc_dst"].data_ptr(),
        part_p, st["scratch"].data_ptr(), st["gbar"].data_ptr(),
        st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
    )

def _pf4_dsort(stream, st):
    # Same argument list as _pf3_dsort plus the per-CTA histogram table appended at the tail.
    fn_pf4_dsort.launch(
        (PF4_DSORT_GRID,), (256,), 0, stream,
        st["recv_eid_p"], st["recv_wgt_p"], st["sc_stage_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), st["sc_dst"].data_ptr(),
        part_p, st["scratch"].data_ptr(), st["gbar"].data_ptr(),
        st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
        st["hcnt"].data_ptr(),
    )

def _pf5_qpush(stream, st):
    fn_pf5_qpush.launch(
        (PF5_QPUSH_GRID,), (256,), 0, stream,
        my_ids_p, my_wgt_p, hidden.data_ptr(),
        st["a_ll_p"], st["dest_counter_p"], st["rows_done_p"],
        st["pushed_count"].data_ptr(), st["push_count"].data_ptr(),
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        st["grid_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX,
    )

def _pf5_dsort(stream, st):
    fn_pf5_dsort.launch(
        (PF5_DSORT_GRID,), (256,), 0, stream,
        st["a_ll_p"], st["rows_done_p"], st["dest_counter_p"],
        st["row_cursor"].data_ptr(), st["a_dst_u8"].data_ptr(),
        st["sc_stage"].data_ptr(), st["recv_eid"].data_ptr(),
        st["recv_wgt"].data_ptr(), st["pull_stage"].data_ptr(),
        st["pull_cnt"].data_ptr(), sti.data_ptr(), swt.data_ptr(),
        sei.data_ptr(), nvi.data_ptr(), pull_ptr.data_ptr(),
        pull_src.data_ptr(), st["sc_dst"].data_ptr(), part_p,
        st["scratch"].data_ptr(), st["gbar"].data_ptr(),
        st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
        st["hcnt"].data_ptr(),
    )

def _pf6_mega(stream, st):
    # v2 descriptor ABI: three kernargs (exp_55 SGPR-spill gate fix). The 256x256 launch is
    # fixed: counted grid barriers require the complete persistent grid to be resident.
    fn_pf6_mega.launch(
        (256,), (256,), 0, stream,
        st["desc"].data_ptr(),
        pperr.data_ptr(), SPIN,
    )

def _pf6c_mega(stream, st):
    # exp_56 Option C: identical three-kernarg ABI, its own 52-slot descriptor (desc_c).
    fn_pf6c_mega.launch(
        (256,), (256,), 0, stream,
        st["desc_c"].data_ptr(),
        pperr.data_ptr(), SPIN,
    )

def _pf6gm_mega(stream, st):
    # exp_59: identical three-kernarg ABI, its own 55-slot descriptor (desc_gm) whose extra
    # slots (53/54) are the device-built expert-aligned tile table + count.
    fn_pf6gm_mega.launch(
        (256,), (256,), 0, stream,
        st["desc_gm"].data_ptr(),
        pperr.data_ptr(), SPIN,
    )

def _pf6mps_mega(stream, st):
    if os.environ.get("K0_MPS_DESC_DUMP") and not getattr(_pf6mps_mega, "_dumped", False):
        _pf6mps_mega._dumped = True
        with open("/out/mps_desc_rank{}.txt".format(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "x")), "w") as _fh:
            _fh.write(repr(st["desc_mps"].detach().cpu().tolist()))
    if not os.environ.get("K0_MPS_SKIP_LAUNCH"):
        fn_pf6mps_mega.launch(
            (256,), (256,), 0, stream,
            st["desc_mps"].data_ptr(),
            pperr.data_ptr(), SPIN,
        )

def _pf3_mega(stream, st):
    fn_pf3_mega.launch(
        (PF3_MEGA_GRID,), (256,), 0, stream,
        my_ids_p, my_wgt_p, hidden.data_ptr(),
        st["a_dst_p"], st["sc_stage_p"], st["recv_eid_p"], st["recv_wgt_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), st["sc_dst"].data_ptr(),
        part_p, st["scratch"].data_ptr(), st["gbar"].data_ptr(),
        st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
    )

def _n2_pf3_nozero(stream, st):
    # PF3's destination sort clears exactly the live partial rows. Do not reintroduce the
    # 587 MiB capacity clear that the package is specifically designed to remove.
    k0_n2.n2_phase1(
        st["a_dst"], st["sc_dst"], w13, fc1_scale, sti1, sei1, nvi1,
        _n2_bufs["a2q"], _n2_bufs["dq2"], stream,
    )
    k0_n2.n2_phase2(
        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
        sti1, swt1, sei1, nvi1, part, stream,
    )

def frozen_body(stream):        # (b) both barriers, both pulls
    _plan(stream); _quant(stream); _gather_pull(stream); _n1g(stream); _combine_pull(stream)

def frozen_n2_body(stream):     # (n2) frozen pipeline + n2 GEMM + STANDALONE barrier2
    _plan(stream); _quant(stream); _gather_pull(stream); _n2(stream); _combine_pull(stream)

def pf_gather_body(stream):
    _plan(stream); _quant(stream); _gather_pf(stream); _n2(stream); _combine_pull(stream)

def pf_combine_body(stream):
    _plan(stream); _quant(stream); _gather_pull(stream); _n2(stream); _combine_pf(stream)

def pf_plan_body(stream):
    _plan_pf(stream); _quant(stream); _gather_pull(stream); _n2(stream); _combine_pull(stream)

def pf_full_body(stream):
    _plan_pf(stream); _quant_pf(stream); _gather_pf(stream); _n2(stream); _combine_pf(stream)

def pf2_plan_body(stream):
    _plan_pf2(stream); _quant_pf(stream); _gather_pf(stream); _n2(stream); _combine_pf(stream)

def pf2_push_body(stream):
    _plan_pf(stream); _quant_pf(stream); _dispatch_pf2(stream, fused=False); _n2_pf2(stream); _combine_pf(stream)

def pf2_full_body(stream):
    _plan_pf2(stream); _quant_pf(stream); _dispatch_pf2(stream, fused=False); _n2_pf2(stream); _combine_pf(stream)

def pf2_fused_body(stream):
    _plan_pf2(stream); _quant_pf(stream); _dispatch_pf2(stream, fused=True); _n2_pf2(stream); _combine_pf(stream)

def pf2_n2r_body(stream):
    # Fully folded decode composition intended by Kimi: dispatch completion is folded into
    # push_fused/transpose_wait and combine completion is folded into n2r phase2/combine_wait.
    _plan_pf2(stream); _quant_pf(stream); _dispatch_pf2(stream, fused=True)
    part[:T_LOC_MAX].zero_()
    k0_n2.n2_phase1(
        a_dst_pf2, sc_dst_pf2, w13, fc1_scale, sti1, sei1, nvi1,
        _n2_bufs["a2q"], _n2_bufs["dq2"], stream,
    )
    fn_n2r_p2.launch(
        (256,), (256,), 0, stream,
        _n2_bufs["a2q"].data_ptr(), _n2_bufs["dq2"].data_ptr(),
        w2c.data_ptr(), fc2_scale.data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        part_p, pf2_n2r_grid.data_ptr(), pf2_n2r_flags_p, rank, WORLD,
    )
    fn_combine_wait.launch(
        (T,), (CBLK,), 0, stream,
        cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
        stage.data_ptr(), pf2_n2r_flags_p, pf2_n2r_grid.data_ptr(),
        pperr.data_ptr(), SPIN, T, H, rank, WORLD,
    )

def pf3_pd_body(stream):
    st = pf3_state["pf3_pd"]
    _pf3_qpush(stream, st)
    _pf3_dsort(stream, st)
    _n2_pf3_nozero(stream, st)
    _combine_pf(stream)

def pf4h_body(stream):
    # pf3_pd's composition with the hierarchical-histogram destination sort swapped in.
    st = pf3_state["pf4h"]
    _pf3_qpush(stream, st)
    _pf4_dsort(stream, st)
    _n2_pf3_nozero(stream, st)
    _combine_pf(stream)

def pf5_ll_body(stream):
    # Doorbell-free LL128 qpush+streaming dsort; N2/barrier2/combine stay identical to PF4H.
    st = pf5_state
    _pf5_qpush(stream, st)
    _pf5_dsort(stream, st)
    _n2_pf3_nozero(stream, st)
    _combine_pf(stream)

def pf6_mega_body(stream):
    # One authored kernel owns dispatch, sort/plan, both N2 phases, and final combine.
    _pf6_mega(stream, pf6_state)

def pf6c_mega_body(stream):
    # exp_56 Option C: same full-region megakernel, chunk-counted release replaces per-packet flags.
    _pf6c_mega(stream, pf6_state)

_pf6gm_g_checked = [False]
def pf6gm_mega_body(stream):
    # exp_59: same full-region megakernel, N2 expert GEMM tiled at M=32*G (G-stacked 32-blocks).
    _pf6gm_mega(stream, pf6_state)
    # exp_59 safety: on the first invocation, read back the COMPILED G the kernel stamped into
    # num_tiles[1] and fail closed if it disagrees with the host PF6GM_G. A silent kernel/host G
    # mismatch already produced one contaminated run; convert it into a loud error.
    if not _pf6gm_g_checked[0]:
        torch.cuda.current_stream().synchronize()
        _compiled_g = int(pf6_state["num_tiles"][1].item())
        _nt = int(pf6_state["num_tiles"][0].item())
        R["pf6gm"]["compiled_G"] = _compiled_g
        R["pf6gm"]["num_tiles_built"] = _nt
        if _compiled_g != PF6GM_G:
            raise RuntimeError(
                f"pf6gm G mismatch: kernel compiled K0P6GM_G={_compiled_g} but host "
                f"PF6GM_G={PF6GM_G}. The frozen runner does not forward K0_PF6GM_G; the "
                f"source default governs the container build. Refusing to time a "
                f"secretly-different-G arm."
            )
        _pf6gm_g_checked[0] = True

_pf6mps_g_checked = [False]
def pf6mps_mega_body(stream):
    global _tbo16_launches
    _tbo16_launches += 1
    if os.environ.get("K0_MPS_TRACE"):
        with open("/out/progress_rank{}.log".format(int(os.environ.get("RANK", "-1"))), "a") as _pf:
            _pf.write("pf6mps_mega_body enter" + chr(10))
    _pf6mps_mega(stream, pf6_state)
    if os.environ.get("K0_MPS_TRACE"):
        with open("/out/progress_rank{}.log".format(int(os.environ.get("RANK", "-1"))), "a") as _pf:
            _pf.write("pf6mps_mega_body launch-returned" + chr(10))
    if not _pf6mps_g_checked[0]:
        torch.cuda.current_stream().synchronize()
        _compiled_g = int(pf6_state["num_tiles"][1].item())
        _nt = int(pf6_state["num_tiles"][0].item())
        R["pf6mps"]["compiled_G"] = _compiled_g
        R["pf6mps"]["num_tiles_built"] = _nt
        if _compiled_g != PF6MPS_G:
            raise RuntimeError(
                f"mps_mega G mismatch: kernel compiled K0P6GM_G={_compiled_g}, "
                f"host PF6GM_G={PF6MPS_G}, N2GM_G={PF6MPS_G}"
            )
        _pf6mps_g_checked[0] = True

def pf3_mega_body(stream):
    st = pf3_state["pf3_mega"]
    _pf3_mega(stream, st)
    _n2_pf3_nozero(stream, st)
    _combine_pf(stream)

# ---- pf3 diagnostics: timestamped twins (DIAGNOSTIC ONLY). The twin launches are identical to
# the candidate launches plus the trailing stamp buffers; the buffers are module globals the
# diagnostic block assigns before first use. ----
_pf3_qts_buf = None
_pf3_dts_buf = None

def _pf3_qpush_ts(stream, st):
    fn_pf3_qpush_ts.launch(
        (PF3_QPUSH_GRID,), (256,), 0, stream,
        my_ids_p, my_wgt_p, hidden.data_ptr(),
        st["a_dst_p"], st["sc_stage_p"], st["recv_eid_p"], st["recv_wgt_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        st["grid_count"].data_ptr(), pperr.data_ptr(),
        T, TOPK, E, rank, WORLD, T_LOC_MAX,
        _pf3_qts_buf.data_ptr(),
    )

def _pf3_dsort_ts(stream, st):
    # K0_PF4_TS switches this to the k0pf4_dsort_prod twin. Its argument list is _pf4_dsort's
    # exactly, plus the trailing stamp buffer — so the twin is a drop-in and the surrounding
    # diagnostic block (eager self-consistency, graph capture, replay, pperr gating) is shared
    # verbatim between the two schemas. Deliberately NOT a new arm name: K0_ARMS silently drops
    # unknown names (§L90), so a diagnostic that must never be skipped is selected by its own
    # env flag and fails loudly below if its module did not load.
    if _K0_PF4_TS:
        fn_pf4_dsort_ts.launch(
            (PF4_DSORT_GRID,), (256,), 0, stream,
            st["recv_eid_p"], st["recv_wgt_p"], st["sc_stage_p"],
            st["dest_counter_p"], st["doorbell_p"],
            st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
            sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
            pull_ptr.data_ptr(), pull_src.data_ptr(), st["sc_dst"].data_ptr(),
            part_p, st["scratch"].data_ptr(), st["gbar"].data_ptr(),
            st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
            T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
            st["hcnt"].data_ptr(),
            _pf3_dts_buf.data_ptr(),
        )
        return
    fn_pf3_dsort_ts.launch(
        (PF3_DSORT_GRID,), (256,), 0, stream,
        st["recv_eid_p"], st["recv_wgt_p"], st["sc_stage_p"],
        st["dest_counter_p"], st["doorbell_p"],
        st["pull_stage"].data_ptr(), st["pull_cnt"].data_ptr(),
        sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
        pull_ptr.data_ptr(), pull_src.data_ptr(), st["sc_dst"].data_ptr(),
        part_p, st["scratch"].data_ptr(), st["gbar"].data_ptr(),
        st["sort_count"].data_ptr(), pperr.data_ptr(), SPIN,
        T, TOPK, E, rank, WORLD, T_LOC_MAX, PADMAX,
        _pf3_dts_buf.data_ptr(),
    )

def pf3_pd_ts_body(stream, out=None, stage_buf=None):
    st = pf3_state["pf3_pd"]
    _pf3_qpush_ts(stream, st)
    _pf3_dsort_ts(stream, st)
    _n2_pf3_nozero(stream, st)
    ms.shmem_barrier_on_stream(stream)
    fn_combine_pf.launch(
        (1024,), (256,), 0, stream,
        (out if out is not None else cand_out).data_ptr(), part_p,
        pull_ptr.data_ptr(), pull_src.data_ptr(),
        (stage_buf if stage_buf is not None else stage).data_ptr(), T, H, rank,
    )

def pf_poison_control_body(stream):
    # Deliberately omits origin quant. Poisoned a_src/sc_src must therefore reach gather and fail
    # the numerical gate; this proves the 16-replay positive test is sensitive to stale sources.
    _plan_pf(stream); _gather_pf(stream); _n2(stream); _combine_pf(stream)

def pf2_poison_control_body(stream):
    # Same positive control for the source-push path: omit quant, so poison must be delivered.
    _plan_pf2(stream); _dispatch_pf2(stream, fused=False); _n2_pf2(stream); _combine_pf(stream)

def frozen_n2r_body(stream):    # (n2r) barrier1 kept; barrier2 FOLDED into the n2r phase-2 tail
    _plan(stream); _quant(stream); _gather_pull(stream)
    part[:T_LOC_MAX].zero_()
    k0_n2.n2_phase1(a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1, _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
    fn_n2r_p2.launch((256,), (256,), 0, stream,
                     _n2_bufs["a2q"].data_ptr(), _n2_bufs["dq2"].data_ptr(),
                     w2c.data_ptr(), fc2_scale.data_ptr(),
                     sti.data_ptr(), swt.data_ptr(), sei.data_ptr(), nvi.data_ptr(),
                     part_p, n2r_grid.data_ptr(), n2r_flags_p, rank, WORLD)
    fn_combine_wait.launch((T,), (CBLK,), 0, stream,
                           cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
                           stage.data_ptr(), n2r_flags_p, n2r_grid.data_ptr(), pperr.data_ptr(), SPIN,
                           T, H, rank, WORLD)

def disp_push_body(stream):     # (c) push dispatch, frozen combine
    _plan(stream); _quant(stream); _emask(stream); _epoch(stream, gen_do.data_ptr())
    _dispatch_push(stream, gen_do.data_ptr(), df_do_p)                 # PUSH
    _n1g(stream); _combine_pull(stream)

def comb_push_body(stream):     # (d) frozen dispatch, push combine
    _plan(stream); _quant(stream); _emask(stream); _gather_pull(stream); _n1g(stream); _epoch(stream, gen_co.data_ptr())
    _combine_push(stream, gen_co.data_ptr(), cf_co_p)                  # PUSH

def full_push_body(stream):     # (e) both boundaries pushed — ZERO exposed host barriers
    _plan(stream); _quant(stream); _emask(stream); _epoch(stream, gen_fp.data_ptr())   # ONE gen bump for BOTH boundaries this replay
    _dispatch_push(stream, gen_fp.data_ptr(), df_fp_p)                 # PUSH
    _n1g(stream)
    _combine_push(stream, gen_fp.data_ptr(), cf_fp_p)                  # PUSH  (same gen: both boundaries, one replay)

# ===================== correctness gate (frozen tolerance, never widened) =====================
def b2f(u16): return (u16.astype(np.uint32) << 16).view(np.float32)
def gate_against_ref(buf, ref_u16):
    c = b2f(buf.view(torch.uint16).cpu().numpy()); ref_f = b2f(ref_u16); d = c - ref_f
    return dict(max_abs=float(np.abs(d).max()), rel_L2=float(np.linalg.norm(d) / np.linalg.norm(ref_f)),
                nonfinite=int((~np.isfinite(c)).sum()), absum=float(np.abs(c).sum()))
def gate(buf): return gate_against_ref(buf, ref)
def ok(g): return bool(g["max_abs"] <= 0.02 and g["rel_L2"] <= 0.01 and g["nonfinite"] == 0 and g["absum"] > 0)
def ok_fp(g): return bool(g["nonfinite"] == 0 and g["absum"] > 0)   # footprint arms: numerics meaningless BY DESIGN; finiteness only
def ok_arm(name, g): return ok_fp(g) if name in _FP_ARMS else ok(g)
_mok_reference_tensor = None
def mok_gate(buf):
    global _mok_reference_tensor
    if _mok_reference_tensor is None:
        _mok_reference_tensor = torch.from_numpy(b2f(ref).copy()).to(
            device=buf.device
        )
    stats = get_mok_error_stats(
        _mok_reference_tensor, buf, distributed=dist
    )
    stats.update(
        absolute_tolerance=MOK_ABSOLUTE_TOLERANCE,
        relative_tolerance=MOK_RELATIVE_TOLERANCE,
        metric="global sum(abs(diff)) / global sum(abs(reference))",
    )
    stats["pass"] = mok_error_passes(
        stats,
        absolute_tolerance=MOK_ABSOLUTE_TOLERANCE,
        relative_tolerance=MOK_RELATIVE_TOLERANCE,
    )
    return stats
def _pair_gate(a, b, live=None):
    # same-run pairwise comparison of two device bf16 tensors over the LIVE prefix [:live] (sorting only
    # guarantees valid/zero output through the live token count; the tail is unspecified). Global max_abs/rel_L2
    # PLUS max per-row rel error with a DENOMINATOR FLOOR (Codex Q2 #2/#3: +1e-12 lets a ~0 ref row dominate;
    # floor at a fraction of the max row norm). nonfinite checks BOTH operands.
    x = a.detach().to(torch.float32).cpu().numpy(); y = b.detach().to(torch.float32).cpu().numpy()
    if live is not None: x, y = x[:live], y[:live]
    x = x.reshape(x.shape[0], -1); y = y.reshape(y.shape[0], -1)
    d = x - y; yn = float(np.linalg.norm(y)); rn = np.linalg.norm(y, axis=1)
    floor = max(1e-6, 1e-3 * float(rn.max()) if rn.size else 1e-6)
    rr = np.linalg.norm(d, axis=1) / np.maximum(rn, floor)
    return dict(max_abs=float(np.abs(d).max()), rel_L2=float(np.linalg.norm(d) / (yn + 1e-12)),
                max_row_rel=float(rr.max()) if rr.size else 0.0,
                nonfinite=int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum()))

# THIS harness = the ON-TARGET compute-only swap. Arms: production (denominator), compute_swap (candidate f
# = MORI dispatch/combine UNCHANGED + n1g exact-FP8 compute), and frozen_pull (custom-transport CONTEXT arm,
# continuity with Stage-1's 1.092x). The push arms are a SEPARATE transport lane, excluded here.
_ALL = {"production": (prod_body, base_out),
        "compute_swap": (compute_swap_body, None),
        "frozen_pull": (frozen_body, cand_out)}
if _HAS_N2: _ALL["compute_swap_n2"] = (compute_swap_n2_body, None)   # exp_47 two-phase write-once arm
if _HAS_N2: _ALL["frozen_n2"] = (frozen_n2_body, None)   # e004: frozen pipeline + n2 GEMM + barrier2
if _HAS_N2:
    _ALL["pf_gather"] = (pf_gather_body, None)
    _ALL["pf_combine"] = (pf_combine_body, None)
    _ALL["pf_plan"] = (pf_plan_body, None)
    _ALL["pf_full"] = (pf_full_body, None)
if _HAS_N2 and _HAS_PF2_PLAN and _PF2_PUSH_OK:
    _ALL["pf2_plan"] = (pf2_plan_body, None)
    _ALL["pf2_push"] = (pf2_push_body, None)
    _ALL["pf2_full"] = (pf2_full_body, None)
    _ALL["pf2_fused"] = (pf2_fused_body, None)
    if _N2R_OK:
        _ALL["pf2_n2r"] = (pf2_n2r_body, None)
if _HAS_N2 and _N2R_OK: _ALL["frozen_n2r"] = (frozen_n2r_body, None)   # exp_47-r folded-rendezvous region arm
if PF3_REQUESTED and not _HAS_N2:
    raise RuntimeError(f"PF3 prefill arms require k0_n2: {_N2_ERR}")
if PF3_REQUESTED and PF3_OK and _HAS_N2:
    if "pf3_pd" in PF3_REQUESTED_ARMS:
        _ALL["pf3_pd"] = (pf3_pd_body, None)
    if "pf3_mega" in PF3_REQUESTED_ARMS:
        _ALL["pf3_mega"] = (pf3_mega_body, None)
    if PF4H_REQUESTED:
        if fn_pf4_dsort is None:
            raise RuntimeError("pf4h requested but k0pf4_dsort did not load")
        _ALL["pf4h"] = (pf4h_body, None)
R["pf3"]["arms_registered"] = [nm for nm in PF3_ARM_NAMES if nm in _ALL]
if PF5_REQUESTED and not _HAS_N2:
    raise RuntimeError(f"PF5 prefill arm requires k0_n2: {_N2_ERR}")
if PF5_REQUESTED and PF5_OK and _HAS_N2:
    if fn_pf5_qpush is None or fn_pf5_dsort is None:
        raise RuntimeError("pf5_ll requested but LL128 kernels did not load")
    _ALL["pf5_ll"] = (pf5_ll_body, None)
R["pf5"]["arms_registered"] = [nm for nm in PF5_ARM_NAMES if nm in _ALL]
if PF6_REQUESTED and not _HAS_N2:
    raise RuntimeError(
        f"PF6 prefill arm requires k0_n2 for the unchanged harness controls: {_N2_ERR}"
    )
if PF6_REQUESTED_ARMS and PF6_OK and _HAS_N2:
    if fn_pf6_mega is None:
        raise RuntimeError("pf6_mega requested but the megakernel did not load")
    if "a2q" not in pf6_state or "dq2" not in pf6_state:
        raise RuntimeError("pf6_mega requested but its private N2 buffers are unavailable")
    _ALL["pf6_mega"] = (pf6_mega_body, None)
R["pf6"]["arms_registered"] = [nm for nm in PF6_ARM_NAMES if nm in _ALL]
# exp_56 Option C chunk-counted-release arm.
if PF6C_REQUESTED and PF6C_OK and _HAS_N2:
    if fn_pf6c_mega is None:
        raise RuntimeError("pf6c_mega requested but the megakernel did not load")
    if "a2q" not in pf6_state or "dq2" not in pf6_state:
        raise RuntimeError("pf6c_mega requested but its private N2 buffers are unavailable")
    if "desc_c" not in pf6_state:
        raise RuntimeError("pf6c_mega requested but its descriptor was not built")
    _ALL["pf6c_mega"] = (pf6c_mega_body, None)
R["pf6c"]["arms_registered"] = [nm for nm in PF6C_ARM_NAMES if nm in _ALL]
if PF6GM_REQUESTED and PF6GM_OK and _HAS_N2:
    if fn_pf6gm_mega is None:
        raise RuntimeError("pf6gm_mega requested but the megakernel did not load")
    if "a2q" not in pf6_state or "dq2" not in pf6_state:
        raise RuntimeError("pf6gm_mega requested but its private N2 buffers are unavailable")
    if "desc_gm" not in pf6_state:
        raise RuntimeError("pf6gm_mega requested but its descriptor was not built")
    _ALL["pf6gm_mega"] = (pf6gm_mega_body, None)
R["pf6gm"]["arms_registered"] = [nm for nm in PF6GM_ARM_NAMES if nm in _ALL]
if PF6MPS_REQUESTED and PF6MPS_OK and _HAS_N2:
    if fn_pf6mps_mega is None or fn_pf6mps_heap_snapshot is None:
        raise RuntimeError("mps_mega requested but its modules did not load")
    for required_key in (
        "a2q", "dq2", "desc_mps", "mps_slots", "mps_heap_descriptor"
    ):
        if required_key not in pf6_state:
            raise RuntimeError(
                f"mps_mega requested but pf6_state[{required_key!r}] is unavailable"
            )
    _ALL["mps_mega"] = (pf6mps_mega_body, None)
R["pf6mps"]["arms_registered"] = [
    nm for nm in PF6MPS_ARM_NAMES if nm in _ALL
]
if K0D_REQUESTED and not _HAS_N2:
    raise RuntimeError(f"k0d decode arms require k0_n2: {_N2_ERR}")
if K0D_REQUESTED and K0D_OK and _HAS_N2:
    _ALL["k0d_s"] = (k0d_s_body, None)
    _ALL["k0d_sq"] = (k0d_sq_body, None)
    _ALL["k0d_sq_b2f"] = (k0d_sq_b2f_body, None)
    _ALL["k0d_f"] = (k0d_f_body, None)
    _ALL["k0d_pd"] = (k0d_pd_body, None)
    _ALL["k0d_mega"] = (k0d_mega_body, None)
R["k0d"]["arms_registered"] = [
    nm for nm in ("k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f", "k0d_pd", "k0d_mega")
    if nm in _ALL
]
if _HAS_C32: _ALL["compute_swap_c32"] = (compute_swap_c32_body, None)   # c32 tail-aware arm (same-session A/B)
if _HAS_AH:  _ALL["compute_swap_ah"]  = (compute_swap_ah_body, None)    # AH W13 address-hoist arm (same-session A/B)
if _HAS_ECS: _ALL["compute_swap_ecs"] = (compute_swap_ecs_body, None)   # ECS full-run co-scheduling arm (lower-bound; AGPR drain)
if _HAS_ECS_D2: _ALL["compute_swap_ecs_d2"] = (compute_swap_ecs_d2_body, None)   # ECS-D2 DECISIVE co-scheduling arm (same-session A/B)
if _HAS_FP:                                                             # weight-footprint 2x2 (Codex 019f8db1)
    _ALL["fp_n1g_real"]    = (fp_n1g_real_body, None)
    _ALL["fp_n1g_alias"]   = (fp_n1g_alias_body, None)
    _ALL["fp_aiter_real"]  = (fp_aiter_real_body, None)
    _ALL["fp_aiter_alias"] = (fp_aiter_alias_body, None)
R["has_c32"] = _HAS_C32; R["c32_import_err"] = _C32_ERR
R["has_n2"] = _HAS_N2; R["n2_import_err"] = _N2_ERR
R["has_ah"] = _HAS_AH; R["ah_import_err"] = _AH_ERR
R["has_ecs"] = _HAS_ECS; R["ecs_import_err"] = _ECS_ERR
R["has_ecs_d2"] = _HAS_ECS_D2; R["ecs_d2_import_err"] = _ECS_D2_ERR
# DEFAULT = genuine 2-arm headline (Codex Q2 #4): custom-transport frozen_pull is a SEPARATE context run
# (K0_ARMS=production,compute_swap,frozen_pull) so its cache/collective footprint never perturbs the A/F number.
_REQUESTED = [nm for nm in os.environ.get("K0_ARMS", "production,frozen_n2,frozen_n2r").split(",") if nm]
_UNKNOWN = [nm for nm in _REQUESTED if nm not in _ALL]
if _UNKNOWN:
    raise RuntimeError(
        f"requested unavailable arms: {_UNKNOWN}; available={sorted(_ALL)} "
        f"pf2_plan={_HAS_PF2_PLAN} pf2_push={_PF2_PUSH_OK}"
    )
_SEL = _REQUESTED
ARMS = [(nm, _ALL[nm][0], _ALL[nm][1]) for nm in _SEL]
R["arms"] = [a[0] for a in ARMS]
_CUSTOM_PLAN_ARMS = {"frozen_pull", "frozen_n2", "frozen_n2r",
                     "pf_gather", "pf_combine", "pf_plan", "pf_full",
                     "pf2_plan", "pf2_push", "pf2_full", "pf2_fused", "pf2_n2r",
                     "k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f"}

# ===================== exact primitive gates (before eager/graph/timing) =====================
def _all_ranks_pass(local_ok):
    x = torch.tensor(int(local_ok), dtype=torch.int32, device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.MIN)
    return bool(x.item())

R["k0pf_primitives"] = {}
hbarrier()
_quant(sp()); torch.cuda.synchronize()
_q_bytes = a_src_u8.clone(); _q_scale = sc_src.clone()
_quant_pf(sp()); torch.cuda.synchronize()
_qd = int((a_src_u8 != _q_bytes).sum().item())
_qsd = int((sc_src != _q_scale).sum().item())
_qok = _all_ranks_pass(_qd == 0 and _qsd == 0)
R["k0pf_primitives"]["quant"] = dict(pass_all_ranks=_qok, byte_diffs=_qd, scale_diffs=_qsd)
if rank == 0:
    print(f"[K0PF GATE] quant_exact pass={_qok} byte_diffs={_qd} scale_diffs={_qsd}", flush=True)
hbarrier()
if not _qok:
    raise RuntimeError("k0pf quant exact-equivalence gate failed; gather/combine were not launched")
del _q_bytes, _q_scale

# Source now holds k0pf quant output, proven byte-identical to frozen.
_gather_pull(sp()); torch.cuda.synchronize()
_g_bytes = a_dst_u8[:T_loc].clone()
_g_scale = sc_dst.reshape(-1)[:NG * T_loc].clone()
_gather_pf(sp()); torch.cuda.synchronize()
_gd = int((a_dst_u8[:T_loc] != _g_bytes).sum().item())
_gsd = int((sc_dst.reshape(-1)[:NG * T_loc] != _g_scale).sum().item())
_gok = _all_ranks_pass(_gd == 0 and _gsd == 0)
R["k0pf_primitives"]["gather"] = dict(pass_all_ranks=_gok, byte_diffs=_gd, scale_diffs=_gsd)
if rank == 0:
    print(f"[K0PF GATE] gather_exact pass={_gok} byte_diffs={_gd} scale_diffs={_gsd}", flush=True)
hbarrier()
if not _gok:
    raise RuntimeError("k0pf gather exact-equivalence gate failed; combine was not launched")

if _PF2_PUSH_OK:
    a_dst_pf2_u8.fill_(0x7E)
    sc_dst_pf2.fill_(float("nan"))
    torch.cuda.synchronize()
    _dispatch_pf2(sp(), fused=False)
    torch.cuda.synchronize()
    _p2d = int((a_dst_pf2_u8[:T_loc] != _g_bytes).sum().item())
    _p2sd = int(
        (sc_dst_pf2.reshape(-1)[: NG * T_loc] != _g_scale).sum().item()
    )
    _p2ok = _all_ranks_pass(_p2d == 0 and _p2sd == 0)
    R["k0pf2_primitives"] = {
        "push": dict(
            pass_all_ranks=_p2ok,
            byte_diffs=_p2d,
            scale_diffs=_p2sd,
            push_grid=PF2_PUSH_GRID,
            transpose_grid=PF2_TRANSPOSE_GRID,
        )
    }
    if rank == 0:
        print(
            f"[K0PF2 GATE] push_exact pass={_p2ok} byte_diffs={_p2d} "
            f"scale_diffs={_p2sd} grids={PF2_PUSH_GRID}/{PF2_TRANSPOSE_GRID}",
            flush=True,
        )
    hbarrier()
    if not _p2ok:
        raise RuntimeError("k0pf2 push exact-equivalence gate failed")

    # Negative control for the folded wait: advertise epoch 1 in a dedicated grid counter but
    # never signal its dedicated symmetric flags. The bounded wait must fail closed via pperr.
    pf2_flags_ctrl, pf2_flags_ctrl_p = mori_t((WORLD, 1), "int32")
    pf2_flags_ctrl.zero_()
    pf2_grid_ctrl = torch.full(
        (1,), PF2_PUSH_GRID, dtype=torch.int32, device="cuda"
    )
    pperr.zero_()
    torch.cuda.synchronize()
    fn_sc_transpose_wait_pf2.launch(
        (PF2_TRANSPOSE_GRID,), (256,), 0, sp(),
        sc_stage_pf2_p, sc_dst_pf2.data_ptr(), nvi.data_ptr(),
        pf2_flags_ctrl_p, pf2_grid_ctrl.data_ptr(), pperr.data_ptr(),
        int(os.environ.get("K0_SPIN_LIMIT_CTRL", "200000")),
        PF2_PUSH_GRID, T_LOC_MAX, rank, WORLD,
    )
    torch.cuda.synchronize()
    _p2_wait_err = int(pperr.cpu().numpy().reshape(-1)[0])
    _p2_wait_ctrl = _all_ranks_pass(_p2_wait_err != 0)
    R["k0pf2_primitives"]["wait_control"] = dict(
        pass_all_ranks=_p2_wait_ctrl, pperr=_p2_wait_err
    )
    if rank == 0:
        print(
            f"[K0PF2 GATE] never_signaled_wait pass={_p2_wait_ctrl} "
            f"pperr={_p2_wait_err}",
            flush=True,
        )
    hbarrier()
    if not _p2_wait_ctrl:
        raise RuntimeError("k0pf2 folded wait negative control did not fail closed")
    pperr.zero_()
    torch.cuda.synchronize()
del _g_bytes, _g_scale

# Use the real n2 partials and real pull plan; only the combine load schedule differs.
_n2(sp()); _combine_pull(sp()); torch.cuda.synchronize()
_c_frozen = cand_out.clone()
_combine_pf(sp()); torch.cuda.synchronize()
_cd = int((cand_out.view(torch.uint16) != _c_frozen.view(torch.uint16)).sum().item())
_cok = _all_ranks_pass(_cd == 0)
R["k0pf_primitives"]["combine"] = dict(pass_all_ranks=_cok, bf16_bit_diffs=_cd)
if rank == 0:
    print(f"[K0PF GATE] combine_bit_exact pass={_cok} bf16_bit_diffs={_cd}", flush=True)
hbarrier()
if not _cok:
    raise RuntimeError("k0pf combine bit-equivalence gate failed; no region arms were launched")
del _c_frozen

# PF3 intentionally changes arrival and sorted-row order, so byte-equality with the frozen plan is
# not meaningful. Validate the invariant sets that n2/combine consume, plus byte-exact fused quant
# on self-destination rows where the producer can name both the source token and allocated row.
_PF3_SELECTED = [
    nm for nm in PF3_ARM_NAMES + PF5_ARM_NAMES if nm in R["arms"]
]
R["pf3_primitives"] = dict(enabled=bool(_PF3_SELECTED), arms={})
if _PF3_SELECTED:
    _quant_pf(sp()); torch.cuda.synchronize()
    _pf3_src_ids = all_ids.reshape(-1).cpu().numpy()
    _pf3_src_wgt = all_wgt.reshape(-1).cpu().numpy()
    _pf3_lo, _pf3_hi = rank * E, (rank + 1) * E
    _pf3_expected_counts = np.asarray([
        int(np.count_nonzero(_pf3_src_ids == e)) for e in range(_pf3_lo, _pf3_hi)
    ], dtype=np.int64)
    _pf3_expected_padded = int(np.sum(((_pf3_expected_counts + 31) // 32) * 32))
    _pf3_ids_local = topk_ids.cpu().numpy()
    _pf3_expected_fanout = np.asarray([
        len(set((int(e) // E) for e in row)) for row in _pf3_ids_local
    ], dtype=np.int32)

    for _pf3_nm in _PF3_SELECTED:
        _st3 = pf3_state[_pf3_nm]
        _st3["a_dst_u8"].fill_(0x7E)
        _st3["sc_stage"].fill_(float("nan"))
        _st3["recv_eid"].fill_(-1)
        _st3["recv_wgt"].fill_(float("nan"))
        _st3["pull_stage"].fill_(-1)
        _st3["pull_cnt"].fill_(-1)
        nvi.zero_(); pull_ptr.zero_(); pull_src.fill_(-1); pperr.zero_()
        torch.cuda.synchronize(); hbarrier()
        if _pf3_nm == "pf3_pd":
            _pf3_qpush(sp(), _st3); _pf3_dsort(sp(), _st3)
        elif _pf3_nm == "pf4h":
            _pf3_qpush(sp(), _st3); _pf4_dsort(sp(), _st3)
        elif _pf3_nm == "pf5_ll":
            _pf5_qpush(sp(), _st3)
            torch.cuda.synchronize()
            _pf5_watermark = int(_st3["dest_counter"].item())
            _pf5_ptrs = {
                name: int(tensor.data_ptr())
                for name, tensor in (
                    ("a_dst", _st3["a_dst_u8"]),
                    ("sc_stage", _st3["sc_stage"]),
                    ("recv_eid", _st3["recv_eid"]),
                    ("recv_wgt", _st3["recv_wgt"]),
                )
            }
            if _pf5_watermark < 0 or _pf5_watermark > T_LOC_MAX or not all(_pf5_ptrs.values()):
                raise RuntimeError(
                    f"PF5 qpush produced invalid watermark/pointers: "
                    f"watermark={_pf5_watermark} ptrs={_pf5_ptrs}"
                )
            if rank == 0:
                print(
                    f"[K0PF5 GATE] qpush completed; watermark={_pf5_watermark} "
                    f"ptrs={_pf5_ptrs}; entering dsort",
                    flush=True,
                )
            _pf5_dsort(sp(), _st3)
        else:
            _pf3_mega(sp(), _st3)
        torch.cuda.synchronize()

        _pf3_nvi = [int(x) for x in nvi.reshape(-1).cpu().numpy()]
        _pf3_tloc = _pf3_nvi[1]
        _pf3_pull_n = int(pull_ptr[T].item())
        _pf3_perr = int(pperr.item())
        _pf3_recv_ids = _st3["recv_eid"][:_pf3_tloc].cpu().numpy()
        _pf3_recv_wgt = _st3["recv_wgt"][:_pf3_tloc].cpu().numpy()
        _pf3_actual_counts = np.asarray([
            int(np.count_nonzero(_pf3_recv_ids == e)) for e in range(_pf3_lo, _pf3_hi)
        ], dtype=np.int64)
        _pf3_count_diff = int(np.abs(_pf3_actual_counts - _pf3_expected_counts).sum())
        _pf3_weight_bit_diffs = 0
        for _e3 in range(_pf3_lo, _pf3_hi):
            _exp3 = np.sort(_pf3_src_wgt[_pf3_src_ids == _e3].view(np.uint32))
            _act3 = np.sort(_pf3_recv_wgt[_pf3_recv_ids == _e3].view(np.uint32))
            if _exp3.shape != _act3.shape:
                _pf3_weight_bit_diffs += abs(int(_exp3.size) - int(_act3.size)) + max(
                    int(_exp3.size), int(_act3.size)
                )
            else:
                _pf3_weight_bit_diffs += int(np.count_nonzero(_exp3 != _act3))

        _pf3_ps = _st3["pull_stage"].reshape(T, TOPK, 2)
        _pf3_self = torch.nonzero(_pf3_ps[:, :, 0] == rank, as_tuple=False)
        if _pf3_self.numel():
            _pf3_tau = _pf3_self[:, 0]
            _pf3_rows = _pf3_ps[_pf3_tau, _pf3_self[:, 1], 1].to(torch.long)
            _pf3_quant_bytes = int(
                (_st3["a_dst_u8"][_pf3_rows] != a_src_u8[_pf3_tau]).sum().item()
            )
            _pf3_quant_scales = int(
                (_st3["sc_stage"][_pf3_rows] != sc_src[_pf3_tau]).sum().item()
            )
        else:
            _pf3_quant_bytes = _pf3_quant_scales = -1
        _pf3_fanout_diff = int(
            np.abs(_st3["pull_cnt"].cpu().numpy() - _pf3_expected_fanout).sum()
        )
        _pf3_local = bool(
            _pf3_perr == 0
            and _pf3_nvi == [_pf3_expected_padded, T_loc]
            and _pf3_pull_n == _pull_n
            and _pf3_count_diff == 0
            and _pf3_weight_bit_diffs == 0
            and _pf3_quant_bytes == 0
            and _pf3_quant_scales == 0
            and _pf3_fanout_diff == 0
        )
        _pf3_all = _all_ranks_pass(_pf3_local)
        R["pf3_primitives"]["arms"][_pf3_nm] = dict(
            pass_all_ranks=_pf3_all, local_pass=_pf3_local, pperr=_pf3_perr,
            nvi=_pf3_nvi, expected_nvi=[_pf3_expected_padded, T_loc],
            pull_n=_pf3_pull_n, expected_pull_n=_pull_n,
            expert_count_abs_diff=_pf3_count_diff,
            weight_bit_diffs=_pf3_weight_bit_diffs,
            self_quant_rows=int(_pf3_self.shape[0]),
            quant_byte_diffs=_pf3_quant_bytes, quant_scale_diffs=_pf3_quant_scales,
            fanout_abs_diff=_pf3_fanout_diff,
        )
        if rank == 0:
            print(
                f"[K0PF3 GATE] {_pf3_nm} pass={_pf3_all} pperr={_pf3_perr} "
                f"nvi={_pf3_nvi}/{[_pf3_expected_padded, T_loc]} "
                f"pull_n={_pf3_pull_n}/{_pull_n} count_diff={_pf3_count_diff} "
                f"weight_diff={_pf3_weight_bit_diffs} self_quant="
                f"{_pf3_quant_bytes}/{_pf3_quant_scales} fanout_diff={_pf3_fanout_diff}",
                flush=True,
            )
        hbarrier()
    R["pf3_primitives"]["pass_all_ranks"] = all(
        x["pass_all_ranks"] for x in R["pf3_primitives"]["arms"].values()
    )
    if PF5_REQUESTED and int(os.environ.get("K0_PF5_PRIMITIVE_ONLY", "0")):
        hbarrier()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(0)

# k0d's three primitive proofs are deliberately run on the real B64 route and real n2 partials,
# before any eager body or graph is allowed to execute.  The normal k0pf primitives above remain
# unchanged and continue to validate their own lane.
_K0D_SELECTED = [
    nm for nm in ("k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f") if nm in R["arms"]
]
R["k0d_primitives"] = dict(enabled=bool(_K0D_SELECTED), arms=_K0D_SELECTED)
if _K0D_SELECTED:
    # (1) fused route + plan + quant must match frozen plan and e22 quant byte-for-byte.
    hbarrier(); perr.zero_(); pperr.zero_()
    _quant(sp()); torch.cuda.synchronize()
    _k0d_q_ref = a_src_u8.clone(); _k0d_sc_ref = sc_src.clone()
    _k0d_fused(sp(), folded=False); torch.cuda.synchronize()
    _k0d_plan_diffs = {
        "own": int((own.reshape(-1) != fr_own.reshape(-1)).sum().item()),
        "tof": int((tof.reshape(-1) != fr_tof.reshape(-1)).sum().item()),
        "tloc": int((tloc.reshape(-1) != fr_tloc.reshape(-1)).sum().item()),
        "gath_live": int((gath[:T_loc] != fr_gath[:T_loc]).sum().item()),
        "sti_live": int((sti[:_padded] != fr_sti[:_padded]).sum().item()),
        "swt_live": int((swt[:_padded] != fr_swt[:_padded]).sum().item()),
        "sei_live": int((sei[:_padded // 32] != fr_sei[:_padded // 32]).sum().item()),
        "nvi": int((nvi != fr_nvi).sum().item()),
        "pull_ptr": int((pull_ptr != fr_pull_ptr).sum().item()),
        "pull_src_live": int((pull_src[:_pull_n] != fr_pull_src[:_pull_n]).sum().item()),
        "quant_bytes": int((a_src_u8 != _k0d_q_ref).sum().item()),
        "quant_scales": int((sc_src != _k0d_sc_ref).sum().item()),
    }
    _k0d_fused_perr = int(perr.item()); _k0d_fused_pperr = int(pperr.item())
    _k0d_fused_local = bool(
        _k0d_fused_perr == 0 and _k0d_fused_pperr == 0
        and all(v == 0 for v in _k0d_plan_diffs.values())
    )
    _k0d_fused_ok = _all_ranks_pass(_k0d_fused_local)
    R["k0d_primitives"]["fused_plan_quant"] = dict(
        pass_all_ranks=_k0d_fused_ok, local_pass=_k0d_fused_local,
        diffs=_k0d_plan_diffs, plan_err=_k0d_fused_perr, pperr=_k0d_fused_pperr,
    )
    if rank == 0:
        print(f"[K0D GATE] fused_plan_quant pass={_k0d_fused_ok} diffs={_k0d_plan_diffs} "
              f"perr={_k0d_fused_perr} pperr={_k0d_fused_pperr}", flush=True)
    hbarrier()
    if not _k0d_fused_ok:
        raise RuntimeError("k0d fused plan+quant primitive gate failed")
    del _k0d_q_ref, _k0d_sc_ref

    # (2) k0d_gather_zero must preserve k0pf's gathered bytes/scales and clear precisely live part rows.
    fn_gather.launch((T_LOC_MAX,), (GBLK,), 0, sp(), a_dst_u8.data_ptr(), sc_dst.data_ptr(),
                     a_src_p, sc_src_p, gath.data_ptr(), nvi.data_ptr(), sc_stage.data_ptr(),
                     T, T_LOC_MAX, K, rank)
    torch.cuda.synchronize()
    _k0d_g_ref = a_dst_u8[:T_loc].clone()
    _k0d_gs_ref = sc_dst.reshape(-1)[:NG * T_loc].clone()
    part[:T_loc].fill_(1.0); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
    _k0d_gather_zero(sp(), folded=False); torch.cuda.synchronize()
    _k0d_gather_d = int((a_dst_u8[:T_loc] != _k0d_g_ref).sum().item())
    _k0d_gather_sd = int((sc_dst.reshape(-1)[:NG * T_loc] != _k0d_gs_ref).sum().item())
    _k0d_part_zero = bool((part[:T_loc].view(torch.uint8) == 0).all().item())
    _k0d_gather_pperr = int(pperr.item())
    _k0d_gather_local = bool(
        _k0d_gather_d == 0 and _k0d_gather_sd == 0 and _k0d_part_zero and _k0d_gather_pperr == 0
    )
    _k0d_gather_ok = _all_ranks_pass(_k0d_gather_local)
    R["k0d_primitives"]["gather_zero"] = dict(
        pass_all_ranks=_k0d_gather_ok, local_pass=_k0d_gather_local,
        byte_diffs=_k0d_gather_d, scale_diffs=_k0d_gather_sd,
        part_live_zero=_k0d_part_zero, pperr=_k0d_gather_pperr,
    )
    if rank == 0:
        print(f"[K0D GATE] gather_zero pass={_k0d_gather_ok} bytes={_k0d_gather_d} "
              f"scales={_k0d_gather_sd} part_zero={_k0d_part_zero} pperr={_k0d_gather_pperr}", flush=True)
    hbarrier()
    if not _k0d_gather_ok:
        raise RuntimeError("k0d gather+zero primitive gate failed")
    del _k0d_g_ref, _k0d_gs_ref

    # (3) Produce real n2 partials after the folded clear, then compare k0d's fixed 16x256 combine
    # body directly against the established k0pf combine.  No synthetic producer payload is accepted here.
    _n2_nozero(sp()); torch.cuda.synchronize(); hbarrier()
    cand_out.zero_(); fn_combine_pf.launch((1024,), (256,), 0, sp(), cand_out.data_ptr(), part_p,
                                           pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(), T, H, rank)
    torch.cuda.synchronize(); _k0d_combine_ref = cand_out.clone()
    cand_out.zero_(); pperr.zero_(); _k0d_combine(sp(), folded=False); torch.cuda.synchronize()
    _k0d_combine_d = int((cand_out.view(torch.uint16) != _k0d_combine_ref.view(torch.uint16)).sum().item())
    _k0d_combine_pperr = int(pperr.item())
    _k0d_combine_local = bool(_k0d_combine_d == 0 and _k0d_combine_pperr == 0)
    _k0d_combine_ok = _all_ranks_pass(_k0d_combine_local)
    R["k0d_primitives"]["combine_real_n2_partial"] = dict(
        pass_all_ranks=_k0d_combine_ok, local_pass=_k0d_combine_local,
        bf16_bit_diffs=_k0d_combine_d, pperr=_k0d_combine_pperr, grid=(16, 256),
    )
    if rank == 0:
        print(f"[K0D GATE] combine_real_n2_partial pass={_k0d_combine_ok} "
              f"bf16_diffs={_k0d_combine_d} pperr={_k0d_combine_pperr}", flush=True)
    hbarrier()
    if not _k0d_combine_ok:
        raise RuntimeError("k0d combine real-n2-partial primitive gate failed")
    del _k0d_combine_ref
    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== event-based per-stage attribution (outside timed graphs) =====================
if int(os.environ.get("K0_STAGE_PROFILE", "1")):
    _stage_names = ("plan", "quant", "b1", "gather", "zero", "n2_p1", "n2_p2", "b2", "combine")
    _stage_reps = int(os.environ.get("K0_STAGE_REPS", "5"))
    R["stage_profile"] = {}
    for _stage_arm in ("frozen_n2", "pf_full"):
        _samples = []
        hbarrier()
        for _rep in range(_stage_reps):
            _ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in _stage_names]
            _st = sp()
            _ev[0][0].record()
            (_plan_pf(_st) if _stage_arm == "pf_full" else _plan(_st))
            _ev[0][1].record()
            _ev[1][0].record()
            (_quant_pf(_st) if _stage_arm == "pf_full" else _quant(_st))
            _ev[1][1].record()
            _ev[2][0].record(); ms.shmem_barrier_on_stream(_st); _ev[2][1].record()
            _ev[3][0].record()
            if _stage_arm == "pf_full":
                fn_gather_pf.launch((2048,), (256,), 0, _st, a_dst_u8.data_ptr(), sc_dst.data_ptr(),
                                    a_src_p, sc_src_p, gath.data_ptr(), nvi.data_ptr(),
                                    sc_stage.data_ptr(), T, T_LOC_MAX, K, rank)
            else:
                fn_gather.launch((T_LOC_MAX,), (GBLK,), 0, _st, a_dst_u8.data_ptr(), sc_dst.data_ptr(),
                                 a_src_p, sc_src_p, gath.data_ptr(), nvi.data_ptr(),
                                 sc_stage.data_ptr(), T, T_LOC_MAX, K, rank)
            _ev[3][1].record()
            _ev[4][0].record(); part[:T_LOC_MAX].zero_(); _ev[4][1].record()
            _ev[5][0].record()
            k0_n2.n2_phase1(a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                            _n2_bufs["a2q"], _n2_bufs["dq2"], _st)
            _ev[5][1].record()
            _ev[6][0].record()
            k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                            sti1, swt1, sei1, nvi1, part, _st)
            _ev[6][1].record()
            _ev[7][0].record(); ms.shmem_barrier_on_stream(_st); _ev[7][1].record()
            _ev[8][0].record()
            if _stage_arm == "pf_full":
                fn_combine_pf.launch((1024,), (256,), 0, _st, cand_out.data_ptr(), part_p,
                                     pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(),
                                     T, H, rank)
            else:
                fn_combine.launch((T,), (CBLK,), 0, _st, cand_out.data_ptr(), part_p,
                                  pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(),
                                  T, H, rank)
            _ev[8][1].record()
            torch.cuda.synchronize()
            _samples.append([s.elapsed_time(e) * 1e3 for s, e in _ev])
            hbarrier()
        _local = np.asarray(_samples, dtype=np.float64)
        _mx = torch.tensor(_local, dtype=torch.float64, device=dev)
        dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
        _maxrank = _mx.cpu().numpy()
        _remote_gather_rows = int((gath[:T_loc, 0] != rank).sum().item())
        _remote_combine_rows = int((pull_src[:_pull_n, 0] != rank).sum().item())
        R["stage_profile"][_stage_arm] = dict(
            reps=_stage_reps,
            local_us_p50={n: float(np.median(_local[:, i])) for i, n in enumerate(_stage_names)},
            maxrank_us_p50={n: float(np.median(_maxrank[:, i])) for i, n in enumerate(_stage_names)},
            local_remote_gather_bytes=_remote_gather_rows * (K + NG * 4),
            local_remote_combine_bytes=_remote_combine_rows * H * 2,
        )
        if rank == 0:
            print(f"[K0PF PROFILE] {_stage_arm} "
                  f"{R['stage_profile'][_stage_arm]['maxrank_us_p50']}", flush=True)

    if PF5_REQUESTED:
        _p5s = ("qpush", "dsort", "n2_p1", "n2_p2", "b2", "combine")
        _samples = []
        hbarrier()
        for _rep in range(_stage_reps):
            _ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in _p5s]
            _st = sp()
            _ev[0][0].record(); _pf5_qpush(_st, pf5_state); _ev[0][1].record()
            _ev[1][0].record(); _pf5_dsort(_st, pf5_state); _ev[1][1].record()
            _ev[2][0].record()
            k0_n2.n2_phase1(
                pf5_state["a_dst"], pf5_state["sc_dst"], w13, fc1_scale,
                sti1, sei1, nvi1, _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _ev[2][1].record()
            _ev[3][0].record()
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _ev[3][1].record()
            _ev[4][0].record(); ms.shmem_barrier_on_stream(_st); _ev[4][1].record()
            _ev[5][0].record(); _combine_pf(_st); _ev[5][1].record()
            torch.cuda.synchronize()
            _samples.append([s.elapsed_time(e) * 1e3 for s, e in _ev])
            hbarrier()
        _local = np.asarray(_samples, dtype=np.float64)
        _mx = torch.tensor(_local, dtype=torch.float64, device=dev)
        dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
        _maxrank = _mx.cpu().numpy()
        R["stage_profile"]["pf5_ll"] = dict(
            reps=_stage_reps,
            local_us_p50={n: float(np.median(_local[:, i])) for i, n in enumerate(_p5s)},
            maxrank_us_p50={
                n: float(np.median(_maxrank[:, i])) for i, n in enumerate(_p5s)
            },
        )
        if rank == 0:
            print(
                f"[K0PF PROFILE] pf5_ll "
                f"{R['stage_profile']['pf5_ll']['maxrank_us_p50']}",
                flush=True,
            )

    # ---- DIAGNOSTIC (Kimi 2026-07-27, prefill megakernel design): pf2_full per-stage attribution,
    # mirroring the loop above but covering the CURRENT best prefill composition (route all-gather,
    # plan v2, quant, b1, push, b_push, transpose, zero, zero-live, n2 p1/p2, b2, combine) plus a
    # same-process production dispatch/GEMM/combine split.  Eager/event-based, outside timed graphs;
    # no effect on gates, arms, or captured bodies.  Prefill regime only.
    if _PF2_PUSH_OK and _HAS_PF2_PLAN and T > 512:
        _p2s = ("route_ag", "plan_v2", "quant", "b1", "push", "b_push", "transpose",
                "zero", "zero_live", "n2_p1", "n2_p2", "b2", "combine")
        _samples = []
        hbarrier()
        for _rep in range(_stage_reps):
            _ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in _p2s]
            _st = sp()
            _ev[0][0].record()
            fn_ag.launch((WORLD,), (256,), 0, _st, all_ids.data_ptr(), all_wgt.data_ptr(),
                         my_ids_p, my_wgt_p, WORLD, T * TOPK, rank)
            _ev[0][1].record()
            _ev[1][0].record()
            k0pf2_plan.build_plan_prod_stream(
                all_ids, all_wgt, own, tof, tloc, tof_part, cnt, erb, ccnt,
                gath, sti, swt, sei, nvi, pull_ptr, pull_src, perr,
                WORLD, T, TOPK, E, rank, T_LOC_MAX, PADMAX, _st,
            )
            _ev[1][1].record()
            _ev[2][0].record(); _quant_pf(_st); _ev[2][1].record()
            _ev[3][0].record(); ms.shmem_barrier_on_stream(_st); _ev[3][1].record()
            _ev[4][0].record()
            fn_push_pf2.launch((PF2_PUSH_GRID,), (256,), 0, _st,
                               a_src_p, sc_src_p, a_dst_pf2_p, sc_stage_pf2_p,
                               own.data_ptr(), tof.data_ptr(), T, WORLD, rank)
            _ev[4][1].record()
            _ev[5][0].record(); ms.shmem_barrier_on_stream(_st); _ev[5][1].record()
            _ev[6][0].record()
            fn_sc_transpose_pf2.launch((PF2_TRANSPOSE_GRID,), (256,), 0, _st,
                                       sc_stage_pf2_p, sc_dst_pf2.data_ptr(), nvi.data_ptr(), T_LOC_MAX)
            _ev[6][1].record()
            _ev[7][0].record(); part[:T_LOC_MAX].zero_(); _ev[7][1].record()
            _ev[8][0].record(); part[:T_loc].zero_(); _ev[8][1].record()
            _ev[9][0].record()
            k0_n2.n2_phase1(a_dst_pf2, sc_dst_pf2, w13, fc1_scale, sti1, sei1, nvi1,
                            _n2_bufs["a2q"], _n2_bufs["dq2"], _st)
            _ev[9][1].record()
            _ev[10][0].record()
            k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                            sti1, swt1, sei1, nvi1, part, _st)
            _ev[10][1].record()
            _ev[11][0].record(); ms.shmem_barrier_on_stream(_st); _ev[11][1].record()
            _ev[12][0].record()
            fn_combine_pf.launch((1024,), (256,), 0, _st, cand_out.data_ptr(), part_p,
                                 pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(),
                                 T, H, rank)
            _ev[12][1].record()
            torch.cuda.synchronize()
            _samples.append([s.elapsed_time(e) * 1e3 for s, e in _ev])
            hbarrier()
        _local = np.asarray(_samples, dtype=np.float64)
        _mx = torch.tensor(_local, dtype=torch.float64, device=dev)
        dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
        _maxrank = _mx.cpu().numpy()
        R["stage_profile"]["pf2_full"] = dict(
            reps=_stage_reps,
            local_us_p50={n: float(np.median(_local[:, i])) for i, n in enumerate(_p2s)},
            maxrank_us_p50={n: float(np.median(_maxrank[:, i])) for i, n in enumerate(_p2s)},
        )
        if rank == 0:
            print(f"[K0PF PROFILE] pf2_full {R['stage_profile']['pf2_full']['maxrank_us_p50']}",
                  flush=True)

    if T > 512:
        # Production dispatch/GEMM/combine totals on the same captured inputs (eager, no graph).
        _prs = ("dispatch", "gemm", "combine")
        _samples = []
        hbarrier()
        for _rep in range(_stage_reps):
            _ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in _prs]
            _st = sp()
            _ev[0][0].record()
            d_a1, d_wgt, d_scale, d_ids, recv = op_prod.dispatch(
                hidden, topk_wgt, None, topk_ids, BLK, -1, WARP)
            _ev[0][1].record()
            _ev[1][0].record()
            _fused = fused_moe(d_a1[:M_TRIM], w1_b, w2_b, d_wgt[:M_TRIM], d_ids[:M_TRIM],
                               expert_mask=expert_mask, activation=ActivationType.Silu,
                               quant_type=QT, w1_scale=fc1_scale, w2_scale=fc2_scale,
                               a1_scale=None, num_local_tokens=recv, doweight_stage1=False,
                               dtype=torch.bfloat16)
            _ev[1][1].record()
            _ev[2][0].record()
            _res = op_prod.combine(_fused, None, topk_ids, BLK, -1, WARP)[0][:T]
            _ev[2][1].record()
            torch.cuda.synchronize()
            _samples.append([s.elapsed_time(e) * 1e3 for s, e in _ev])
            hbarrier()
        _local = np.asarray(_samples, dtype=np.float64)
        _mx = torch.tensor(_local, dtype=torch.float64, device=dev)
        dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
        _maxrank = _mx.cpu().numpy()
        R["stage_profile"]["production"] = dict(
            reps=_stage_reps,
            local_us_p50={n: float(np.median(_local[:, i])) for i, n in enumerate(_prs)},
            maxrank_us_p50={n: float(np.median(_maxrank[:, i])) for i, n in enumerate(_prs)},
        )
        if rank == 0:
            print(f"[K0PF PROFILE] production {R['stage_profile']['production']['maxrank_us_p50']}",
                  flush=True)

    # Decode-only attribution for the two k0d bodies.  This is deliberately eager/event based:
    # it preserves the real cross-rank rendezvous and the exact n2 calls while exposing each
    # kernel's contribution.  Headline A/B timing below remains graph replay and is unaffected.
    if _K0D_SELECTED:
        _k0d_stage_specs = {}
        if "k0d_s" in _K0D_SELECTED:
            _k0d_stage_specs["k0d_s"] = (
                ("fused", "b1", "gather_zero", "n2_p1", "n2_p2", "b2", "combine"),
                (
                    lambda st: _k0d_fused(st, folded=False),
                    lambda st: fn_k0d_barrier.launch(
                        (1,), (64,), 0, st, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                        pperr.data_ptr(), SPIN, rank, WORLD,
                    ),
                    lambda st: _k0d_gather_zero(st, folded=False),
                    lambda st: k0_n2.n2_phase1(
                        a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                        _n2_bufs["a2q"], _n2_bufs["dq2"], st,
                    ),
                    lambda st: k0_n2.n2_phase2(
                        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                        sti1, swt1, sei1, nvi1, part, st,
                    ),
                    lambda st: fn_k0d_barrier.launch(
                        (1,), (64,), 0, st, k0ds_b2_flags_p, k0ds_b2_count.data_ptr(),
                        pperr.data_ptr(), SPIN, rank, WORLD,
                    ),
                    lambda st: _k0d_combine(st, folded=False),
                ),
            )
        if "k0d_sq" in _K0D_SELECTED:
            _k0d_stage_specs["k0d_sq"] = (
                ("fused_plan", "quant", "b1", "gather_zero", "n2_p1", "n2_p2", "b2", "combine"),
                (
                    lambda st: _k0d_fused(st, folded=False, do_quant=False),
                    lambda st: _quant(st),
                    lambda st: fn_k0d_barrier.launch(
                        (1,), (64,), 0, st, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                        pperr.data_ptr(), SPIN, rank, WORLD,
                    ),
                    lambda st: _k0d_gather_zero(st, folded=False),
                    lambda st: k0_n2.n2_phase1(
                        a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                        _n2_bufs["a2q"], _n2_bufs["dq2"], st,
                    ),
                    lambda st: k0_n2.n2_phase2(
                        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                        sti1, swt1, sei1, nvi1, part, st,
                    ),
                    lambda st: fn_k0d_barrier.launch(
                        (1,), (64,), 0, st, k0ds_b2_flags_p, k0ds_b2_count.data_ptr(),
                        pperr.data_ptr(), SPIN, rank, WORLD,
                    ),
                    lambda st: _k0d_combine(st, folded=False),
                ),
            )
        if "k0d_sq_b2f" in _K0D_SELECTED:
            _k0d_stage_specs["k0d_sq_b2f"] = (
                ("fused_plan", "quant", "b1", "gather_zero", "n2_p1", "n2_p2", "combine_b2f"),
                (
                    lambda st: _k0d_fused(st, folded=False, do_quant=False),
                    lambda st: _quant(st),
                    lambda st: fn_k0d_barrier.launch(
                        (1,), (64,), 0, st, k0ds_b1_flags_p, k0ds_b1_count.data_ptr(),
                        pperr.data_ptr(), SPIN, rank, WORLD,
                    ),
                    lambda st: _k0d_gather_zero(st, folded=False),
                    lambda st: k0_n2.n2_phase1(
                        a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                        _n2_bufs["a2q"], _n2_bufs["dq2"], st,
                    ),
                    lambda st: k0_n2.n2_phase2(
                        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                        sti1, swt1, sei1, nvi1, part, st,
                    ),
                    lambda st: _k0d_combine_b2f(st),
                ),
            )
        if "k0d_f" in _K0D_SELECTED:
            _k0d_stage_specs["k0d_f"] = (
                ("fused", "gather_zero", "n2_p1", "n2_p2", "combine"),
                (
                    lambda st: _k0d_fused(st, folded=True),
                    lambda st: _k0d_gather_zero(st, folded=True),
                    lambda st: k0_n2.n2_phase1(
                        a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                        _n2_bufs["a2q"], _n2_bufs["dq2"], st,
                    ),
                    lambda st: k0_n2.n2_phase2(
                        _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                        sti1, swt1, sei1, nvi1, part, st,
                    ),
                    lambda st: _k0d_combine(st, folded=True),
                ),
            )

        for _stage_arm, (_k0d_stage_names, _k0d_stage_calls) in _k0d_stage_specs.items():
            _samples = []
            hbarrier()
            for _rep in range(_stage_reps):
                _st = sp()
                _ev = [(torch.cuda.Event(True), torch.cuda.Event(True))
                       for _ in _k0d_stage_names]
                hbarrier()
                for _i, _stage_call in enumerate(_k0d_stage_calls):
                    _ev[_i][0].record()
                    _stage_call(_st)
                    _ev[_i][1].record()
                torch.cuda.synchronize()
                _samples.append([s.elapsed_time(e) * 1e3 for s, e in _ev])
                hbarrier()
            _local = np.asarray(_samples, dtype=np.float64)
            _mx = torch.tensor(_local, dtype=torch.float64, device=dev)
            dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
            _maxrank = _mx.cpu().numpy()
            R["stage_profile"][_stage_arm] = dict(
                reps=_stage_reps,
                local_us_p50={
                    n: float(np.median(_local[:, i])) for i, n in enumerate(_k0d_stage_names)
                },
                maxrank_us_p50={
                    n: float(np.median(_maxrank[:, i]))
                    for i, n in enumerate(_k0d_stage_names)
                },
            )
            if rank == 0:
                print(f"[K0D PROFILE] {_stage_arm} "
                      f"{R['stage_profile'][_stage_arm]['maxrank_us_p50']}", flush=True)
        perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# Decode-only continuous phase timelines for the push ingress arms. Unlike the older diagnostic
# above, this preserves every boundary from one start event on each rank; offline analysis can
# therefore select the total-critical rank for each replay instead of adding unrelated rank maxima.
_DECODE_PHASE_ARMS = [
    nm for nm in ("production", "k0d_pd", "k0d_mega", "pf3_pd", "pf3_mega", "pf4h")
    if nm in R["arms"]
]
if int(os.environ.get("K0_DECODE_EAGER_PHASE_PROFILE", "0")) and T <= 512 and _DECODE_PHASE_ARMS:
    _decode_phase_reps = int(os.environ.get("K0_DECODE_PHASE_REPS", "100"))
    _decode_profile = dict(
        reps=_decode_phase_reps,
        units="microseconds",
        arms={},
        order_policy="cyclic rotation per replay",
        graph_mode=False,
        purpose="diagnostic continuous eager timeline; uninstrumented graph A/B remains authoritative",
    )
    _decode_stage_names = {
        "production": ("dispatch", "aiter_gemm", "combine"),
        "k0d_pd": ("quant", "push", "dsort", "n2_p1", "n2_p2", "combine_edge_c"),
        "k0d_mega": ("quant", "mega_ingress", "n2_p1", "n2_p2", "combine_edge_c"),
    }
    _decode_samples = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_cumulative = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_pperr = {nm: [] for nm in _DECODE_PHASE_ARMS if nm != "production"}
    _decode_orders = []

    def _decode_profile_once(_nm):
        _st = sp()
        _names = _decode_stage_names[_nm]
        _events = [torch.cuda.Event(True) for _ in range(len(_names) + 1)]
        _events[0].record()
        if _nm == "production":
            d_a1, d_wgt, d_scale, d_ids, recv = op_prod.dispatch(
                hidden, topk_wgt, None, topk_ids, BLK, -1, WARP,
            )
            _events[1].record()
            _fused = fused_moe(
                d_a1[:M_TRIM], w1_b, w2_b, d_wgt[:M_TRIM], d_ids[:M_TRIM],
                expert_mask=expert_mask, activation=ActivationType.Silu, quant_type=QT,
                w1_scale=fc1_scale, w2_scale=fc2_scale, a1_scale=None,
                num_local_tokens=recv, doweight_stage1=False, dtype=torch.bfloat16,
            )
            _events[2].record()
            _res = op_prod.combine(_fused, None, topk_ids, BLK, -1, WARP)[0][:T]
            _events[3].record()
            # Keep graph/API return objects live through synchronization.
            _decode_profile["_production_last_ptr"] = int(_res.data_ptr())
        elif _nm == "k0d_pd":
            pperr.zero_()
            _quant(_st); _events[1].record()
            _k0d_push_ingress(_st); _events[2].record()
            _k0d_dsort_ingress(_st); _events[3].record()
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _events[4].record()
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _events[5].record()
            _k0d_combine_pd(_st); _events[6].record()
        else:
            pperr.zero_()
            _quant(_st); _events[1].record()
            _k0d_mega_ingress(_st); _events[2].record()
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _events[3].record()
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _events[4].record()
            _k0d_combine_pd(_st); _events[5].record()
        torch.cuda.synchronize()
        _cum = [float(_events[0].elapsed_time(e) * 1e3) for e in _events[1:]]
        _dur = [_cum[0]] + [float(_cum[i] - _cum[i - 1]) for i in range(1, len(_cum))]
        return _dur, _cum

    hbarrier()
    for _rep in range(_decode_phase_reps):
        _rotation = _rep % len(_DECODE_PHASE_ARMS)
        _order = _DECODE_PHASE_ARMS[_rotation:] + _DECODE_PHASE_ARMS[:_rotation]
        _decode_orders.append(_order)
        for _nm in _order:
            hbarrier()
            _dur, _cum = _decode_profile_once(_nm)
            _decode_samples[_nm].append(_dur)
            _decode_cumulative[_nm].append(_cum)
            if _nm != "production":
                _decode_pperr[_nm].append(int(pperr.item()))
            hbarrier()

    # Measure only the direct owner-pull accumulation body on the last valid custom partial/CSR.
    # This is deliberately a separate pass so it cannot perturb the continuous main timelines.
    _plain_combine = []
    if any(nm in _DECODE_PHASE_ARMS for nm in ("k0d_pd", "k0d_mega")):
        for _rep in range(_decode_phase_reps):
            hbarrier()
            _s_ev = torch.cuda.Event(True); _e_ev = torch.cuda.Event(True)
            _s_ev.record()
            _k0d_combine(sp(), folded=False)
            _e_ev.record()
            torch.cuda.synchronize()
            _plain_combine.append(float(_s_ev.elapsed_time(_e_ev) * 1e3))
            hbarrier()
    _decode_profile["plain_owner_pull_combine_us"] = _plain_combine

    _decode_profile_local_ok = True
    for _nm in _DECODE_PHASE_ARMS:
        _arr = np.asarray(_decode_samples[_nm], dtype=np.float64)
        _decode_profile["arms"][_nm] = dict(
            stage_names=list(_decode_stage_names[_nm]),
            duration_us=_decode_samples[_nm],
            cumulative_us=_decode_cumulative[_nm],
            local_p50_us={
                n: float(np.median(_arr[:, i]))
                for i, n in enumerate(_decode_stage_names[_nm])
            },
            local_p95_us={
                n: float(np.percentile(_arr[:, i], 95))
                for i, n in enumerate(_decode_stage_names[_nm])
            },
            pperr=_decode_pperr.get(_nm, []),
        )
        if _nm != "production":
            _decode_profile_local_ok = bool(
                _decode_profile_local_ok and all(x == 0 for x in _decode_pperr[_nm])
            )
    _decode_profile["orders"] = _decode_orders
    _decode_profile["local_protocol_ok"] = _decode_profile_local_ok
    _decode_profile["pass_all_ranks"] = _all_ranks_pass(_decode_profile_local_ok)
    R["decode_phase_profile"] = _decode_profile
    if rank == 0:
        print(
            f"[K0D PHASE PROFILE] reps={_decode_phase_reps} "
            f"arms={_DECODE_PHASE_ARMS} local_p50="
            f"{ {nm: _decode_profile['arms'][nm]['local_p50_us'] for nm in _DECODE_PHASE_ARMS} } "
            f"plain_combine_p50={float(np.median(_plain_combine)) if _plain_combine else None}",
            flush=True,
        )
    if not _decode_profile["pass_all_ranks"]:
        raise RuntimeError("decode phase profile observed a protocol error")
    pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== eager correctness on every arm + positive control =====================
R["eager"] = {}
_eager_all_local = True
# exp_32 P0: the [MOK GATE] loop below runs AFTER this loop has finished, so for
# every candidate arm it re-reads the SAME cand_out bytes -- whichever candidate
# ran last (_obuf returns the identical object for all of them). Compute each
# arm's MoK gate HERE, while its own bytes are still live, and have that loop
# consume the stash. Without this, P1's poison for a given arm is only checked
# when that arm happens to be last in the rotated order.
_mok_eager_stash = {}
for name, body, buf in ARMS:
    if os.environ.get("K0_MPS_TRACE"):
        with open("/out/progress_rank{}.log".format(int(os.environ.get("RANK", "-1"))), "a") as _pf:
            _pf.write("arm " + name + chr(10))
    hbarrier()
    if name not in _PROD_LIKE: _poison_out(name)   # FAIRNESS: never touch production's own combine-output view
    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); body(sp()); torch.cuda.synchronize()
    if name == "mps_mega" and MPS_MODE16:
        # exp_02: mode 16's first combine completes one launch late, so the
        # eager arm needs a second launch before any gate can see epoch 0's
        # output (poisoned in between so the survivor gate stays exact).
        _poison_out(name)   # P4b: arm-aware half_write poison, not the global branch
        body(sp()); torch.cuda.synchronize()
    if name not in _PROD_LIKE:
        R.setdefault("poison", {})[name] = _poison_report("eager", name)
        # exp_32 V2a: prove the detector is wired to a GATE, not just a print.
        # Poison ONE row of an otherwise-correct buffer; mok_gate MUST fail.
        # Restores the row bit-for-bit, so this cannot perturb any later gate.
        if MOK_POISON_OUT and os.environ.get("K0_MOK_POISON_SELFTEST", "1") != "0":
            # [T-1] is the last live row of every arm's own output region
            # (half-0 row T-1 for the deferred mps arm as well)
            _st_idx = ((_tbo16_half_latest() * T + (T - 1))
                       if (MPS_MODE16 and name == "mps_mega") else (T - 1))
            _st_row = cand_out[_st_idx].clone()
            cand_out.view(torch.int16)[_st_idx].fill_(MOK_POISON_I16)
            _st = mok_gate(_obuf(name))
            cand_out[_st_idx].copy_(_st_row)
            R["poison"][name + "_selftest_fails"] = bool(not _st["pass"])
            if rank == 0:
                print(f"[POISON SELFTEST] arm={name} one_row_poisoned_fails="
                      f"{not _st['pass']} nonfinite={_st['nonfinite']} "
                      f"relative={_st['relative_error']}", flush=True)
    if K0_BENCHMARK_PROTOCOL == "mok_eager":
        _mok_eager_stash[name] = mok_gate(_obuf(name))
    R["eager"][name] = gate(_obuf(name)); R["eager"][name]["pass"] = ok_arm(name, R["eager"][name])
    R["eager"][name]["pperr"] = int(pperr.cpu().numpy().reshape(-1)[0])   # !=0 => an acquire timed out (diagnostic)
    R["eager"][name]["plan_err"] = int(perr.cpu().numpy().reshape(-1)[0]) if name in _CUSTOM_PLAN_ARMS else 0
    _eager_local = bool(
        R["eager"][name]["pass"]
        and R["eager"][name]["plan_err"] == 0
        and R["eager"][name]["pperr"] == 0
    )
    R["eager"][name]["local_pass"] = _eager_local
    R["eager"][name]["pass"] = _all_ranks_pass(_eager_local)
    _eager_all_local = bool(_eager_all_local and _eager_local)
    if rank == 0: print(f"[MARK] eager {name} rel_L2={R['eager'][name]['rel_L2']:.5f} pass={R['eager'][name]['pass']} pperr={R['eager'][name]['pperr']}", flush=True)
    # exp_56 pf6c-specific STRUCTURAL control (count/occupancy based; the frozen plan_equivalence
    # gate validates the reference path, NOT any megakernel stage). Element-wise sti/swt comparison
    # is INVALID (scatter order is deliberately atomic-cursor nondeterministic, hkp_sort.hpp:12-13),
    # so this checks the DETERMINISTIC quantities that a dropped/duplicated row (every bug class we
    # hit) would perturb: (1) per-expert live counts vs production, (2) nvi[0] padded total & nvi[1]
    # == T_ext, (3) total live pairs == T_loc*TOPK. (4) per-expert pair-SET identity is a deferred
    # coverage gap (own experiment). This runs right after pf6c's single correctness-gate epoch.
    if name == "pf6c_mega":
        _c = {}
        _pf6c_nvi = pf6_state["nvi"].detach().cpu().numpy().reshape(-1)
        _pf6c_scratch = pf6_state["scratch"].detach().cpu().numpy().reshape(-1)
        _fr_nvi = fr_nvi.detach().cpu().numpy().reshape(-1)
        _fr_cnt = fr_cnt.detach().cpu().numpy().reshape(-1)   # production per-expert LIVE counts [E]
        _T_ext_expect = WORLD * MAXTOK
        # (2) padded total must match production; extent is pf6c-specific = T_ext.
        _c["nvi0_padded_pf6c"] = int(_pf6c_nvi[0]); _c["nvi0_padded_prod"] = int(_fr_nvi[0])
        _c["nvi0_match"] = bool(int(_pf6c_nvi[0]) == int(_fr_nvi[0]))
        _c["nvi1_ext_pf6c"] = int(_pf6c_nvi[1]); _c["nvi1_ext_expect"] = int(_T_ext_expect)
        _c["nvi1_match"] = bool(int(_pf6c_nvi[1]) == int(_T_ext_expect))
        # (1) per-expert live counts: hkp_sort scratch carve OffCnt=80 holds cnt[e] for e in [0,E).
        _OFF_CNT = 80
        _pf6c_cnt = _pf6c_scratch[_OFF_CNT:_OFF_CNT + E].astype("int64")
        _cnt_diffs = int((_pf6c_cnt != _fr_cnt[:E].astype("int64")).sum())
        _c["per_expert_cnt_diffs"] = _cnt_diffs
        _c["per_expert_cnt_match"] = bool(_cnt_diffs == 0)
        # (3) total live pairs == sum of per-expert counts; production reference = fr_cnt sum.
        _c["live_pairs_pf6c"] = int(_pf6c_cnt.sum()); _c["live_pairs_prod"] = int(_fr_cnt[:E].sum())
        _c["live_pairs_match"] = bool(int(_pf6c_cnt.sum()) == int(_fr_cnt[:E].sum()))
        _c_local = bool(_c["nvi0_match"] and _c["nvi1_match"] and
                        _c["per_expert_cnt_match"] and _c["live_pairs_match"])
        _c["pass_all_ranks"] = _all_ranks_pass(_c_local)
        _c["deferred_coverage_gap"] = "per-expert (row,slot) pair-SET identity (control 4) NOT checked"
        # exp_56 spin-distribution diag: [0]=max spins to SUCCESS, [1]=max spins at FAIL (chunk poll).
        _sd = pf6_state["spin_dbg"].detach().cpu().numpy().reshape(-1).tolist()
        _c["chunk_poll_spin_success_max"] = int(_sd[0])
        _c["chunk_poll_spin_fail_max"] = int(_sd[1])
        R["pf6c_structural_control"] = _c
        if rank == 0:
            print(f"[PF6C SPIN] chunk_poll success_max={_sd[0]} fail_max={_sd[1]}", flush=True)
        if rank == 0:
            print(f"[PF6C STRUCT] pass={_c['pass_all_ranks']} nvi0={_c['nvi0_padded_pf6c']}/"
                  f"{_c['nvi0_padded_prod']} nvi1={_c['nvi1_ext_pf6c']}/{_c['nvi1_ext_expect']} "
                  f"cnt_diffs={_c['per_expert_cnt_diffs']} "
                  f"live_pairs={_c['live_pairs_pf6c']}/{_c['live_pairs_prod']}", flush=True)
R["eager_all_pass"] = _all_ranks_pass(_eager_all_local)
R["mok_correctness"] = {}
_mok_correctness_all_local = True
if K0_BENCHMARK_PROTOCOL == "mok_eager":
    for name, _, _ in ARMS:
        _mok_stats = _mok_eager_stash[name]   # exp_32 P0: that arm's OWN bytes
        _mok_local = bool(
            _mok_stats["pass"]
            and R["eager"][name]["plan_err"] == 0
            and R["eager"][name]["pperr"] == 0
        )
        _mok_stats["local_pass"] = _mok_local
        _mok_stats["pass_all_ranks"] = _all_ranks_pass(_mok_local)
        R["mok_correctness"][name] = _mok_stats
        _mok_correctness_all_local = bool(
            _mok_correctness_all_local and _mok_local
        )
        if rank == 0:
            print(
                f"[MOK GATE] {name} max_abs={_mok_stats['abs_error_max']:.6f} "
                f"relative={_mok_stats['relative_error']:.6f} "
                f"pass={_mok_stats['pass_all_ranks']}",
                flush=True,
            )
        # exp_56 diagnostic: localize deterministic corruption by dumping which OUTPUT tokens fail
        # and the (source,row) of their combine primaries. Runs for the pf6c arm only when it fails
        # the gate (rank-0, cheap). The frozen runner does not forward a custom env flag, so this is
        # keyed on the arm name + failing gate rather than an env var.
        if name == "pf6c_mega" and not _mok_stats["pass"]:
            try:
                _dref = _mok_reference_tensor            # [T,H] f32 oracle view
                _dcand = _obuf(name).to(torch.float32)
                _drow_max = (_dcand - _dref).abs().max(dim=1).values
                _dbad = torch.nonzero(_drow_max > 0.1, as_tuple=False).flatten()
                _dn = int(_dbad.numel())
                _dinfo = {"arm": name, "rank": rank, "n_bad_tokens": _dn, "T": int(T),
                          "worst_row_maxabs": float(_drow_max.max().item())}
                if _dn > 0:
                    _didx = _dbad.detach().cpu().tolist()
                    _dinfo["bad_token_first64"] = _didx[:64]
                    _pp = pull_ptr.detach().cpu().flatten().tolist()
                    _ps = pull_src.detach().cpu().tolist()
                    _srcs = {}
                    for _tk in _didx[:512]:
                        for _k in range(_pp[_tk], _pp[_tk + 1]):
                            _pe, _rw = _ps[_k]
                            _srcs.setdefault(int(_pe), []).append(int(_rw))
                    _dinfo["bad_by_source"] = {
                        str(_pe): {"count": len(_r), "row_min": min(_r), "row_max": max(_r),
                                    "local_pos_min": min(_r) % MAXTOK,
                                    "local_pos_max": max(_r) % MAXTOK}
                        for _pe, _r in _srcs.items()}
                R.setdefault("pf6c_diag", []).append(_dinfo)
                # print on EVERY rank when that rank has bad tokens — corruption is rank-asymmetric
                # (rank 0 can be clean while the global gate fails on another rank).
                if _dn > 0:
                    print(f"[PF6C DIAG r{rank}] {json.dumps(_dinfo)[:1400]}", flush=True)
            except Exception as _ediag:
                print(f"[PF6C DIAG r{rank}] failed: {_ediag}", flush=True)
    R["mok_correctness_all_pass"] = _all_ranks_pass(
        _mok_correctness_all_local
    )
else:
    R["mok_correctness_all_pass"] = True

# ===================== K0_PROBE=1: CONTENT probe v3 (early exit) =====================
if os.environ.get("K0_PROBE", "0") == "1":
    if rank == 0:
        import numpy as _np
        # reference: aiter per_1x128 quant of `hidden` DIRECTLY -- row space = local tokens [0,T)
        # num_rows is Optional[Tensor], not int (aiter::per_group_quant_hip); hidden is exactly
        # [T,H], so omitting it quantises all T rows -- the same reference, no tensor needed.
        _ref_fp8, _ref_sc = _hipquant(hidden, scale=None, quant_dtype=w1_b.dtype,
                                      transpose_scale=True)
        refu8 = _ref_fp8.view(torch.uint8).cpu().numpy()
        refsc = _ref_sc.cpu().numpy()
        gu8 = a_dst_u8.cpu().numpy()
        gsc = sc_dst.cpu().numpy()
        gh = gath.cpu().numpy()
        nv = nvi.cpu().numpy().reshape(-1)
        Tloc = int(nv[1])
        print(f"[PROBE3] shapes: Tloc={Tloc} refu8={refu8.shape} refsc={refsc.shape} "
              f"gu8={gu8.shape} gsc={gsc.shape} gath={gh.shape} T={T} rank={rank}", flush=True)
        # refsc is group-major [NG, T]; tolerate the transposed layout rather than crash
        _sc_gm = (refsc.shape[0] == NG)
        def _refsc(kg, tok):
            return float(refsc[kg, tok]) if _sc_gm else float(refsc[tok, kg])
        same_bad = 0; scale_bad = 0; same_tot = 0; rem_tot = 0; skipped = 0
        ex = []; bad_t = []; bad_st = []; worst_sc = 0.0
        for t in range(min(Tloc, gu8.shape[0])):
            sr, st = int(gh[t, 0]), int(gh[t, 1])
            if sr < 0 or st < 0:
                skipped += 1; continue
            if sr != rank:
                rem_tot += 1; continue
            same_tot += 1
            if st >= refu8.shape[0]:
                if len(ex) < 6: ex.append(("st_oob", t, sr, st))
                continue
            if not (gu8[t] == refu8[st]).all():
                same_bad += 1; bad_t.append(t); bad_st.append(st)
                if len(ex) < 6:
                    ex.append(("row", t, sr, st, int((gu8[t] != refu8[st]).sum())))
            for kg in range(NG):
                r_ = _refsc(kg, st); g_ = float(gsc[t, kg])
                d_ = abs(g_ - r_)
                if d_ > worst_sc: worst_sc = d_
                if d_ > 1e-6 * max(1e-3, abs(r_)):
                    scale_bad += 1
                    if len(ex) < 12: ex.append(("scale", t, kg, g_, r_))
                    break
        print(f"[PROBE3] Tloc={Tloc} same_rank_rows={same_tot} remote_rows={rem_tot} unused={skipped} "
              f"same_rank_row_bad={same_bad} scale_bad={scale_bad} worst_scale_delta={worst_sc:.3e}", flush=True)
        print(f"[PROBE3] ex={ex}", flush=True)
        if bad_t:
            bt = _np.array(bad_t); bs = _np.array(bad_st)
            print(f"[PROBE3] clustering: bad_t min={int(bt.min())} max={int(bt.max())} n={bt.size} "
                  f"contiguous={bool(bt.size == int(bt.max())-int(bt.min())+1)} "
                  f"uniform={bool(bt.size == same_tot)} | bad_st min={int(bs.min())} max={int(bs.max())} "
                  f"first12_t={bt[:12].tolist()} first12_st={bs[:12].tolist()}", flush=True)
        else:
            print("[PROBE3] clustering: no bad same-rank rows", flush=True)
    hbarrier()
    raise SystemExit("PROBE3 DONE")

# ---- bridge validation (Codex Q2): production snapshot + matched-pair oracle/inner + false-scale control ----
bridge_val_ok = True
if "compute_swap" in R["arms"]:
    hbarrier()
    # production SNAPSHOT (op.combine returns a VIEW of the persistent output buffer -> clone before reuse).
    torch.cuda.synchronize(); prod_body(sp()); torch.cuda.synchronize(); prod_snap = _ph["prod"].clone()
    # matched-pair: fmoe (oracle, correct scale) + n1g on IDENTICAL sort/quant, one dispatch.
    hbarrier(); mp = bridge_matched_diag(sp(), transpose_scale=True); LIVE = mp["live"]; R["live_tokens"] = LIVE
    R["bridge_oracle"]      = gate(mp["res_fmoe"]); R["bridge_oracle"]["pass"] = ok(R["bridge_oracle"])   # fmoe(my twin) vs corpus
    R["compute_swap_final"] = gate(mp["res_n1g"]);  R["compute_swap_final"]["pass"] = ok(R["compute_swap_final"])  # n1g vs corpus
    R["oracle_vs_prod"]     = _pair_gate(mp["res_fmoe"], prod_snap)                 # TWIN PROOF: fmoe(my sort+quant)==production
    R["inner_n1g_vs_fmoe"]  = _pair_gate(mp["buf_n1g"], mp["buf_fmoe"], live=LIVE)  # COMPUTE ISOLATION (row-aligned, live prefix)
    R["final_n1g_vs_fmoe"]  = _pair_gate(mp["res_n1g"], mp["res_fmoe"])
    _op = R["oracle_vs_prod"]; _in = R["inner_n1g_vs_fmoe"]
    R["oracle_vs_prod"]["pass"]    = bool(_op["max_abs"] <= 0.02 and _op["rel_L2"] <= 0.01 and _op["nonfinite"] == 0)
    R["inner_n1g_vs_fmoe"]["pass"] = bool(_in["max_abs"] <= 0.02 and _in["rel_L2"] <= 0.01 and _in["nonfinite"] == 0)
    # false-scale POSITIVE CONTROL: n1g fed token-major scale MUST fail BOTH the inner (vs correct fmoe) and final gate.
    hbarrier(); mpc = bridge_matched_diag(sp(), transpose_scale=False)
    _ci = _pair_gate(mpc["buf_n1g"], mpc["buf_fmoe"], live=mpc["live"])
    R["bridge_control_inner"] = _ci; R["bridge_control_final"] = gate(mpc["res_n1g"])
    R["bridge_control_inner_fails"] = bool(_ci["rel_L2"] > 0.01 or _ci["max_abs"] > 0.02)
    R["bridge_control_final_fails"] = bool(not ok(R["bridge_control_final"]))
    # The token-major-scale control reliably fails the SENSITIVE inner (pre-combine, per-token) gate — that is
    # the correct bridge positive control: it proves the inner n1g-vs-fmoe gate is non-vacuous AND sensitive to
    # the group-major scale contract. The final combine can MASK it (finding: real activation per-128-block
    # scales cluster near amax/448 ~ near-uniform, so a scale-LAYOUT swap is a weak post-reduction perturbation).
    # The FINAL corpus gate's non-vacuity is proven independently by the generic frozen-combine control_fails.
    R["bridge_control_fails"] = bool(R["bridge_control_inner_fails"])
    bridge_val_ok = bool(R["bridge_oracle"]["pass"] and R["compute_swap_final"]["pass"] and
                         R["oracle_vs_prod"]["pass"] and R["inner_n1g_vs_fmoe"]["pass"] and R["bridge_control_fails"])
    R["bridge_val_ok"] = bridge_val_ok
    if rank == 0:
        print(f"[MARK] LIVE={LIVE} bridge_oracle pass={R['bridge_oracle']['pass']} rel_L2={R['bridge_oracle']['rel_L2']:.5f} "
              f"| cswap_final pass={R['compute_swap_final']['pass']} rel_L2={R['compute_swap_final']['rel_L2']:.5f}", flush=True)
        print(f"[MARK] oracle_vs_prod rel_L2={_op['rel_L2']:.6f} max_abs={_op['max_abs']:.5f} pass={R['oracle_vs_prod']['pass']} "
              f"| inner n1g_vs_fmoe rel_L2={_in['rel_L2']:.5f} max_row_rel={_in['max_row_rel']:.5f} pass={R['inner_n1g_vs_fmoe']['pass']}", flush=True)
        print(f"[MARK] false_scale_control inner_fails={R['bridge_control_inner_fails']} (inner rel_L2={_ci['rel_L2']:.5f}) "
              f"final_fails={R['bridge_control_final_fails']} (final rel_L2={R['bridge_control_final']['rel_L2']:.5f}; masked by combine) "
              f"-> bridge_control_fails={R['bridge_control_fails']} (MUST be True) | bridge_val_ok={bridge_val_ok}", flush=True)
    hbarrier()

# ---- n2 compute-isolation gate: n2 inner == fmoe inner on IDENTICAL sort/quant (definitive) ----
R["n2_gate_ok"] = True
if "compute_swap_n2" in R["arms"] and _HAS_N2:
    hbarrier(); mn2 = n2_matched_diag(sp())
    R["n2_final_vs_corpus"] = gate(mn2["res_n2"]); R["n2_final_vs_corpus"]["pass"] = ok(R["n2_final_vs_corpus"])
    R["n2_inner_vs_fmoe"] = _pair_gate(mn2["buf_n2"], mn2["buf_fmoe"], live=mn2["live"])
    _n2i = R["n2_inner_vs_fmoe"]
    R["n2_inner_vs_fmoe"]["pass"] = bool(_n2i["max_abs"] <= 0.02 and _n2i["rel_L2"] <= 0.01 and _n2i["nonfinite"] == 0)
    R["n2_gate_ok"] = bool(R["n2_final_vs_corpus"]["pass"] and R["n2_inner_vs_fmoe"]["pass"])
    if rank == 0:
        print(f"[MARK] n2_final_vs_corpus pass={R['n2_final_vs_corpus']['pass']} rel_L2={R['n2_final_vs_corpus']['rel_L2']:.5f} "
              f"| n2_inner_vs_fmoe rel_L2={_n2i['rel_L2']:.6f} max_row_rel={_n2i['max_row_rel']:.6f} max_abs={_n2i['max_abs']:.6f} "
              f"pass={R['n2_inner_vs_fmoe']['pass']} | n2_gate_ok={R['n2_gate_ok']}", flush=True)
    hbarrier()

# ---- c32 compute-isolation gate: c32 inner == oracle n1g inner on IDENTICAL sort/quant (definitive) ----
R["c32_gate_ok"] = True
if "compute_swap_c32" in R["arms"] and _HAS_C32:
    hbarrier(); mc = c32_matched_diag(sp())
    R["c32_final_vs_corpus"] = gate(mc["res_c32"]); R["c32_final_vs_corpus"]["pass"] = ok(R["c32_final_vs_corpus"])
    R["c32_inner_vs_oracle"] = _pair_gate(mc["buf_c32"], mc["buf_oracle"], live=mc["live"])
    _c32i = R["c32_inner_vs_oracle"]
    R["c32_inner_vs_oracle"]["pass"] = bool(_c32i["max_abs"] <= 0.02 and _c32i["rel_L2"] <= 0.01 and _c32i["nonfinite"] == 0)
    R["c32_final_vs_oracle"] = _pair_gate(mc["res_c32"], mc["res_oracle"])
    R["c32_gate_ok"] = bool(R["c32_final_vs_corpus"]["pass"] and R["c32_inner_vs_oracle"]["pass"])
    if rank == 0:
        print(f"[MARK] c32_final_vs_corpus pass={R['c32_final_vs_corpus']['pass']} rel_L2={R['c32_final_vs_corpus']['rel_L2']:.5f} "
              f"| c32_inner_vs_oracle rel_L2={_c32i['rel_L2']:.6f} max_row_rel={_c32i['max_row_rel']:.6f} max_abs={_c32i['max_abs']:.6f} "
              f"pass={R['c32_inner_vs_oracle']['pass']} | c32_gate_ok={R['c32_gate_ok']}", flush=True)
    hbarrier()

# ---- AH compute-isolation gate: ah inner == oracle n1g inner on IDENTICAL sort/quant (definitive) ----
# The W13 address hoist streams the SAME bytes as the oracle, so its MFMA inputs are BIT-IDENTICAL; the
# inner pair-gate should be near-zero (only BF16-combine nondeterminism), a STRONGER equality than c32's skip.
R["ah_gate_ok"] = True
if "compute_swap_ah" in R["arms"] and _HAS_AH:
    hbarrier(); mah = ah_matched_diag(sp())
    R["ah_final_vs_corpus"] = gate(mah["res_ah"]); R["ah_final_vs_corpus"]["pass"] = ok(R["ah_final_vs_corpus"])
    R["ah_inner_vs_oracle"] = _pair_gate(mah["buf_ah"], mah["buf_oracle"], live=mah["live"])
    _ahi = R["ah_inner_vs_oracle"]
    R["ah_inner_vs_oracle"]["pass"] = bool(_ahi["max_abs"] <= 0.02 and _ahi["rel_L2"] <= 0.01 and _ahi["nonfinite"] == 0)
    R["ah_final_vs_oracle"] = _pair_gate(mah["res_ah"], mah["res_oracle"])
    R["ah_gate_ok"] = bool(R["ah_final_vs_corpus"]["pass"] and R["ah_inner_vs_oracle"]["pass"])
    if rank == 0:
        print(f"[MARK] ah_final_vs_corpus pass={R['ah_final_vs_corpus']['pass']} rel_L2={R['ah_final_vs_corpus']['rel_L2']:.5f} "
              f"| ah_inner_vs_oracle rel_L2={_ahi['rel_L2']:.6f} max_row_rel={_ahi['max_row_rel']:.6f} max_abs={_ahi['max_abs']:.6f} "
              f"pass={R['ah_inner_vs_oracle']['pass']} | ah_gate_ok={R['ah_gate_ok']}", flush=True)
    hbarrier()

# ---- ECS compute-isolation gate: ecs inner == oracle n1g inner on IDENTICAL sort/quant (definitive) ----
# ecs only REORDERS which CTA runs each (block,g); the per-block GEMM math is byte-for-byte the oracle's, so the
# inner pair-gate should be near-zero (only BF16-combine atomic-order nondeterminism), the same strong equality ah has.
R["ecs_gate_ok"] = True
if "compute_swap_ecs" in R["arms"] and _HAS_ECS:
    hbarrier(); mecs = ecs_matched_diag(sp())
    R["ecs_final_vs_corpus"] = gate(mecs["res_ecs"]); R["ecs_final_vs_corpus"]["pass"] = ok(R["ecs_final_vs_corpus"])
    R["ecs_inner_vs_oracle"] = _pair_gate(mecs["buf_ecs"], mecs["buf_oracle"], live=mecs["live"])
    _ecsi = R["ecs_inner_vs_oracle"]
    R["ecs_inner_vs_oracle"]["pass"] = bool(_ecsi["max_abs"] <= 0.02 and _ecsi["rel_L2"] <= 0.01 and _ecsi["nonfinite"] == 0)
    R["ecs_final_vs_oracle"] = _pair_gate(mecs["res_ecs"], mecs["res_oracle"])
    R["ecs_gate_ok"] = bool(R["ecs_final_vs_corpus"]["pass"] and R["ecs_inner_vs_oracle"]["pass"])
    if rank == 0:
        print(f"[MARK] ecs_final_vs_corpus pass={R['ecs_final_vs_corpus']['pass']} rel_L2={R['ecs_final_vs_corpus']['rel_L2']:.5f} "
              f"| ecs_inner_vs_oracle rel_L2={_ecsi['rel_L2']:.6f} max_row_rel={_ecsi['max_row_rel']:.6f} max_abs={_ecsi['max_abs']:.6f} "
              f"pass={R['ecs_inner_vs_oracle']['pass']} | ecs_gate_ok={R['ecs_gate_ok']}", flush=True)
    hbarrier()

# ---- ECS-D2 compute-isolation gate: ecs_d2 inner == oracle n1g inner on IDENTICAL sort/quant (definitive) ----
R["ecs_d2_gate_ok"] = True
if "compute_swap_ecs_d2" in R["arms"] and _HAS_ECS_D2:
    hbarrier(); md2 = ecs_d2_matched_diag(sp())
    R["ecs_d2_final_vs_corpus"] = gate(md2["res_ecs_d2"]); R["ecs_d2_final_vs_corpus"]["pass"] = ok(R["ecs_d2_final_vs_corpus"])
    R["ecs_d2_inner_vs_oracle"] = _pair_gate(md2["buf_ecs_d2"], md2["buf_oracle"], live=md2["live"])
    _d2i = R["ecs_d2_inner_vs_oracle"]
    R["ecs_d2_inner_vs_oracle"]["pass"] = bool(_d2i["max_abs"] <= 0.02 and _d2i["rel_L2"] <= 0.01 and _d2i["nonfinite"] == 0)
    R["ecs_d2_final_vs_oracle"] = _pair_gate(md2["res_ecs_d2"], md2["res_oracle"])
    R["ecs_d2_gate_ok"] = bool(R["ecs_d2_final_vs_corpus"]["pass"] and R["ecs_d2_inner_vs_oracle"]["pass"])
    if rank == 0:
        print(f"[MARK] ecs_d2_final_vs_corpus pass={R['ecs_d2_final_vs_corpus']['pass']} rel_L2={R['ecs_d2_final_vs_corpus']['rel_L2']:.5f} "
              f"| ecs_d2_inner_vs_oracle rel_L2={_d2i['rel_L2']:.6f} max_row_rel={_d2i['max_row_rel']:.6f} max_abs={_d2i['max_abs']:.6f} "
              f"pass={R['ecs_d2_inner_vs_oracle']['pass']} | ecs_d2_gate_ok={R['ecs_d2_gate_ok']}", flush=True)
    hbarrier()

# positive control: the frozen combine's store-over-a-peer bug MUST fail the gate (proves the gate can fail).
hbarrier(); cand_out.zero_()
# e004: n1g hardcodes a [4096,3584] BF16 view and rejects prefill shapes. The control tests the
# BUGGY COMBINE, not the GEMM, so n2 serves identically as the partial producer.
_plan(sp()); _quant(sp()); _gather_pull(sp()); _n2(sp())
ms.shmem_barrier_on_stream(sp())
fn_combine_bug.launch((T,), (CBLK,), 0, sp(), cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
                      stage.data_ptr(), T, H, rank)
torch.cuda.synchronize()
# exp_02: under MPS_MODE16 the doubled buffer's live control region is rows
# [0,T) (every arm, including the frozen control body, writes that region).
_ctrl_view = cand_out[0:T] if MPS_MODE16 else cand_out
R["control"] = gate(_ctrl_view); R["control_fails"] = bool(not ok(R["control"]))
if K0_BENCHMARK_PROTOCOL == "mok_eager":
    R["mok_control"] = mok_gate(_ctrl_view)
    R["mok_control_fails"] = bool(not R["mok_control"]["pass"])
else:
    R["mok_control_fails"] = True
hbarrier()
if rank == 0: print(f"[MARK] control_fails={R['control_fails']}", flush=True)

# ===================== Mixture-of-Kittens-style eager benchmark =====================
if K0_BENCHMARK_PROTOCOL == "mok_eager":
    if "production" not in R["arms"] or len(R["arms"]) < 2:
        raise RuntimeError(
            "mok_eager benchmark requires production plus at least one candidate; "
            f"got {R['arms']}"
        )

    def _finish_mok_benchmark(exit_code):
        R["config"] = dict(
            input_mode=K0_INPUT_MODE,
            benchmark_protocol=K0_BENCHMARK_PROTOCOL,
            T=T,
            t_loc_max=T_LOC_MAX,
            padmax=PADMAX,
            block_num=BLK,
            warp_num=WARP,
            arms=R["arms"],
            maxtok=MAXTOK,
            maxtok_prod=MAXTOK_PROD,
            mok_warmup_iters=MOK_WARMUP_ITERS,
            mok_timed_iters=MOK_TIMED_ITERS,
            mok_correctness_metric=(
                "global sum(abs(diff)) / global sum(abs(reference))"
            ),
            mok_absolute_tolerance=MOK_ABSOLUTE_TOLERANCE,
            mok_relative_tolerance=MOK_RELATIVE_TOLERANCE,
            pf6gm_g=PF6GM_G if PF6GM_REQUESTED else None,
            mps_config=PF6MPS_CONFIG,
            mps_soak_iters=(
                int(os.environ.get("K0_MPS_SOAK_ITERS", "600"))
                if PF6MPS_REQUESTED else None
            ),
            pf6_grid=(
                dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS)
                if PF6_REQUESTED else None
            ),
        )
        _source_paths = {
            "executor": os.path.abspath(__file__),
            "correctness": os.path.join(
                _MOK_BENCH_DIR, "correctness.py"
            ),
            "synthetic_inputs": os.path.join(
                _MOK_BENCH_DIR, "synthetic_inputs.py"
            ),
        }
        R["benchmark_source_sha256"] = {}
        for _source_name, _source_path in _source_paths.items():
            with open(_source_path, "rb") as _source_file:
                R["benchmark_source_sha256"][_source_name] = hashlib.sha256(
                    _source_file.read()
                ).hexdigest()
        hbarrier()
        _outdir = os.environ.get("K0_OUT_DIR", OUT)
        os.makedirs(_outdir, exist_ok=True)
        _outjson = f"{_outdir}/k0pf_mok_synthetic_rank{rank}.json"
        with open(_outjson, "w") as _output_file:
            json.dump(R, _output_file, indent=2)
        if rank == 0:
            print(
                f"[MOK SYNTHETIC EAGER] status={R['mok_eager']['status']} "
                f"ratios={json.dumps(R['mok_eager'].get('ratios', {}))} "
                f"gate_ok={R['mok_eager']['timing_gate_ok']}",
                flush=True,
            )
        hbarrier()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(exit_code)

    _mok_pre_timing_local_ok = bool(
        R["mok_correctness_all_pass"]
        and R["mok_control_fails"]
        and R.get("pf3_primitives", {}).get("pass_all_ranks", True)
    )
    _mok_pre_timing_all_ok = _all_ranks_pass(_mok_pre_timing_local_ok)
    if not _mok_pre_timing_all_ok:
        R["mok_eager"] = {
            "status": "blocked_pre_timing_mok_correctness",
            "reference_method": (
                "same-run production with Mixture-of-Kittens FP8 metric/tolerance"
            ),
            "graph_mode": False,
            "warmup_iters": MOK_WARMUP_ITERS,
            "timed_iters": 0,
            "arm_order": list(R["arms"]),
            "arms": {},
            "ratios": {},
            "timing_gate_ok": False,
            "scope": (
                "no latency result: a pre-timing correctness gate failed"
            ),
        }
        R["timing_gate_ok"] = False
        _finish_mok_benchmark(2)

    if "mps_mega" in R["arms"]:
        R["mps_negative_control"] = {
            "kind": "frozen combine store-over-peer correctness control",
            "failed_as_required": bool(R["mok_control_fails"]),
        }
        _mps_soak_iters = int(os.environ.get("K0_MPS_SOAK_ITERS", "600"))
        if _mps_soak_iters != 600:
            raise RuntimeError(
                f"MPS protocol requires exactly 600 soak epochs; got {_mps_soak_iters}"
            )
        _mps_body = next(body for name, body, _ in ARMS if name == "mps_mega")
        pperr.zero_()
        torch.cuda.synchronize()
        hbarrier()
        _mps_soak_error = 0
        _mps_soak_completed = 0
        _mps_soak_poison = 0
        _mps_soak_poison_epoch = -1
        for _mps_epoch in range(_mps_soak_iters):
            # exp_32 P2: re-poison before EVERY epoch. The soak is fully untimed
            # (no HIP events anywhere in this loop), so this turns one
            # end-of-soak check into 600 independent single-epoch coverage trials.
            _poison_out("mps_mega")
            _mps_body(sp())
            torch.cuda.synchronize()
            _mps_local_error = int(pperr.item())
            _mps_local_poison = _poison_count("mps_mega")
            _mps_error_tensor = torch.tensor(
                [_mps_local_error, _mps_local_poison],
                dtype=torch.int32, device="cuda",
            )
            dist.all_reduce(_mps_error_tensor, op=dist.ReduceOp.MAX)
            _mps_soak_error = int(_mps_error_tensor[0].item())
            _mps_soak_poison = int(_mps_error_tensor[1].item())
            _mps_soak_completed = _mps_epoch + 1
            if _mps_soak_error != 0 or _mps_soak_poison != 0:
                if _mps_soak_poison != 0:
                    _mps_soak_poison_epoch = _mps_epoch
                    _poison_report("soak", "mps_mega")
                break
# (juxtaposition preserved from patch 1)
                break
        hbarrier()
        _mps_soak_gate = mok_gate(_obuf("mps_mega"))
        _mps_soak_gate["pperr"] = _mps_soak_error
        _mps_soak_gate["pass_all_ranks"] = _all_ranks_pass(
            bool(_mps_soak_gate["pass"] and _mps_soak_error == 0)
        )
        R["mps_soak"] = {
            "requested_epochs": _mps_soak_iters,
            "completed_epochs": _mps_soak_completed,
            "pperr": _mps_soak_error,
            "poison_survivors": _mps_soak_poison,
            "poison_first_bad_epoch": _mps_soak_poison_epoch,
            "correctness": _mps_soak_gate,
            "pass": bool(
                _mps_soak_completed == _mps_soak_iters
                and _mps_soak_error == 0
                and _mps_soak_poison == 0
                and _mps_soak_gate["pass_all_ranks"]
            ),
        }
        if rank == 0:
            print(
                f"[MPS SOAK] completed={_mps_soak_completed}/{_mps_soak_iters} "
                f"pperr={_mps_soak_error} poison={_mps_soak_poison} "
                f"poison_epoch={_mps_soak_poison_epoch} "
                f"pass={R['mps_soak']['pass']}",
                flush=True,
            )
        # --- exp_05 stage 0 (Distributed-HipKittens overnight): ADDITIVE diagnostic.
        # Dumps the MPS device timestamp block that the kernel already writes and
        # this harness already allocates but never read. Print-only: no arm, no
        # gate, no timing path touches this. Guarded so a scope/shape change can
        # never fail a run. Layout: 8 u32 scalars (= 4 int64) then 8 u64 stamps.
        # After a 600-epoch soak the max-stamps are all from the FINAL epoch and
        # are mutually consistent; the min-stamp (KSTART) is from the first epoch
        # and must NOT be used for a per-epoch delta.
        if rank == 0:
            try:
                _mps_ts_t = None
                try:
                    _mps_ts_t = pf6_state["mps_state"]
                except Exception:
                    _mps_ts_t = _pf6_mps_state
                _w = [int(x) & 0xFFFFFFFFFFFFFFFF
                      for x in _mps_ts_t.detach().cpu().tolist()[4:12]]
                _nm = ["FIRST_READY_inv", "LAST_READY", "DRAIN", "M7_DONE",
                       "REDUCE_DONE", "M5_DONE", "M2_DONE", "M6_DONE"]
                print("[MPS TS] " + " ".join(f"{a}={b}" for a, b in zip(_nm, _w)),
                      flush=True)
                _m2, _m6, _m7, _rd, _dr = _w[6], _w[7], _w[3], _w[4], _w[2]
                _m5 = _w[5]
                if _m2 and _m5 and _m6:
                    print(f"[MPS TS SPLIT] plan_M3toM5={_m5 - _m2} "
                          f"M6={_m6 - _m5} (ticks, 1 tick = 0.01 us)", flush=True)
                if _m2 and _m6 and _m7 and _rd:
                    print(f"[MPS TS DELTA] planM6={_m6 - _m2} M7={_m7 - _m6} "
                          f"combine={_rd - _m7} servicedrain={_dr - _m6} "
                          f"m2_to_end={_rd - _m2}", flush=True)
            except Exception as _e:
                print(f"[MPS TS] unavailable: {type(_e).__name__}: {_e}", flush=True)
            # --- exp_10 (Distributed-HipKittens overnight): ADDITIVE diagnostic.
            # spin_dbg[0] = max spins to SUCCESS on M2's per-(source,chunk) poll,
            # [1] = max spins at FAIL. Written by the SAME M2 that pf6gm_mega and
            # mps_mega both run, so this measures the DISPATCH WAIT of the
            # ratchet kernel, not just the MPS arm. Already recorded on device
            # and already printed by the harness -- but only under the pf6c
            # structural-control block, which our arms never run. Print-only.
            try:
                _sdm = pf6_state["spin_dbg"].detach().cpu().numpy().reshape(-1).tolist()
                print(f"[MPS SPIN] chunk_poll success_max={int(_sdm[0])} "
                      f"fail_max={int(_sdm[1])} limit={os.environ.get('K0_SPIN_LIMIT','?')}",
                      flush=True)
            except Exception as _e2:
                print(f"[MPS SPIN] unavailable: {type(_e2).__name__}: {_e2}", flush=True)
        if not R["mps_soak"]["pass"]:
            # Terminal protocol failure: preserve pperr and exit. Never clear it
            # and retry this epoch/configuration.
            R["mok_eager"] = {
                "status": "blocked_pre_timing_mps_soak",
                "reference_method": "same-run production",
                "graph_mode": False,
                "warmup_iters": 0,
                "timed_iters": 0,
                "arm_order": list(R["arms"]),
                "arms": {},
                "ratios": {},
                "timing_gate_ok": False,
                "scope": "no latency result: MPS 600-epoch soak failed",
            }
            R["timing_gate_ok"] = False
            _finish_mok_benchmark(3)

    def _mok_rank_max(samples):
        tensor = torch.tensor(samples, dtype=torch.float64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.cpu().numpy()

    def _mok_measure_arm(name, body):
        if name != "production":
            _poison_out(name)
        perr.zero_()
        pperr.zero_()
        torch.cuda.synchronize()
        hbarrier()
        for _ in range(MOK_WARMUP_ITERS):
            body(sp())
        torch.cuda.synchronize()
        dist.barrier()

        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(MOK_TIMED_ITERS)
        ]
        for start, end in events:
            start.record()
            body(sp())
            end.record()
        torch.cuda.synchronize()
        dist.barrier()

        local_us = np.asarray(
            [start.elapsed_time(end) * 1e3 for start, end in events],
            dtype=np.float64,
        )
        aligned_us = _mok_rank_max(local_us)
        # exp_32 P3: the timed loop leaves cand_out holding 600 epochs of
        # accumulated writes, so a row a LATE epoch failed to write still reads
        # an earlier epoch's bit-identical correct answer and the gates below
        # cannot see it. Re-poison and run ONE untimed epoch so they see
        # single-epoch coverage in the steady state the timed loop just left.
        # This is strictly after every HIP event has been consumed and after the
        # rank-max all-reduce above, so it cannot touch a measurement. pperr is
        # NOT cleared: it is atomicOr-accumulated, so output_gate["pperr"] below
        # still reports the OR over warmup + timed + this epoch. Skipped when
        # pperr is already set, because M8 suppresses every batch on a live error
        # bit and would leave a false poison survivor.
        if MOK_POISON_OUT and name != "production" and int(pperr.item()) == 0:
            _poison_out(name)
            torch.cuda.synchronize()
            hbarrier()
            body(sp())
            torch.cuda.synchronize()
            hbarrier()
            R.setdefault("poison", {})[name + "_post_timing"] = _poison_report(
                "post_timing", name)
        strict_gate = gate(_obuf(name))
        strict_gate["pass"] = ok_arm(name, strict_gate)
        output_gate = mok_gate(_obuf(name))
        output_gate["pperr"] = int(pperr.item())
        output_gate["plan_err"] = (
            int(perr.item()) if name in _CUSTOM_PLAN_ARMS else 0
        )
        output_gate["local_pass"] = bool(
            output_gate["pass"]
            and output_gate["pperr"] == 0
            and output_gate["plan_err"] == 0
        )
        output_gate["pass_all_ranks"] = _all_ranks_pass(output_gate["local_pass"])
        return {
            "local_us": local_us.tolist(),
            "aligned_rank_max_us": aligned_us.tolist(),
            "p50_us": float(np.percentile(aligned_us, 50)),
            "p95_us": float(np.percentile(aligned_us, 95)),
            "correctness": output_gate,
            "strict_k0_correctness_diagnostic": strict_gate,
        }

    _mok_arms = {}
    for _mok_name, _mok_body, _ in ARMS:
        _mok_arms[_mok_name] = _mok_measure_arm(_mok_name, _mok_body)
        hbarrier()
    _mok_prod = _mok_arms["production"]

    # ================= exp_58 Phase 1: per-module residual attribution =================
    # Env-gated, reset-per-launch, cumulative-prefix diagnostic timing of pf6gm_mega.
    # See experiments/exp_58_residual_attribution/design.md for why the shipped
    # debug_stop knob (fault-bisection only) cannot be timed through the frozen runner:
    #  * a truncated kernel fails correctness => the standard path blocks ALL timing;
    #  * truncation skips M9 (counter reset + epoch advance) => a free-running truncated
    #    warmup overflows dest_counter and never advances the epoch.
    # This path resets the FULL-REGION protocol state before EVERY timed launch, so each
    # launch runs REAL work from the epoch-1, zero-skew state the correctness gate validates.
    # It runs AFTER the standard same-run timing above, so it perturbs neither arm.
    if os.environ.get("K0_PF6GM_DECOMP", "0") == "1" and "pf6gm_mega" in R["arms"]:
        _dc_W = int(os.environ.get("K0_PF6GM_DECOMP_WARMUP", "20"))
        _dc_N = int(os.environ.get("K0_PF6GM_DECOMP_TIMED", "40"))
        # symmetric + local full-region state that M9/M0 own; must be epoch-1 clean per launch.
        _dc_state = [
            pf6_state["mega_count"], pf6_state["dest_counter"], pf6_state["pushed_count"],
            pf6_state["qpush_done"], pf6_state["combine_done"], pf6_state["gbar"],
            pf6_state["chunk_ready"], pf6_state["rows_done"], pf6_state["row_ready"],
            pf6_state["retired"], pf6_state["a2_done"], pf6_state["part_done"],
            pf6_state["row_remaining"],
        ]

        def _dc_reset():
            for _t in _dc_state:
                _t.zero_()
            cand_out.zero_()
            pperr.zero_()
            perr.zero_()

        # base desc for pf6gm (slot 49 overwritten per phase). Keep the original for restore.
        _dc_desc_list = pf6_state["desc_gm"].tolist()
        _dc_orig_desc = pf6_state["desc_gm"]

        def _dc_measure_phase(_p):
            _desc_p = list(_dc_desc_list)
            _desc_p[49] = int(_p)
            pf6_state["desc_gm"] = torch.tensor(_desc_p, dtype=torch.int64, device="cuda")
            # warmup: reset + launch each iter (truncated launches skip M9, so a free-run
            # warmup would corrupt state — reset every time). A FULL cross-rank rendezvous
            # (sync + barrier) surrounds every launch: truncated phases write PEER symmetric
            # buffers (a_ll/dest_counter/chunk_ready/rows_done), so no rank may reset those
            # until ALL ranks' previous launch has fully retired. Both barriers sit OUTSIDE
            # the HIP-event window, so they never enter a timed measurement.
            for _ in range(_dc_W):
                _dc_reset()
                torch.cuda.synchronize(); dist.barrier()
                _pf6gm_mega(sp(), pf6_state)
                torch.cuda.synchronize(); dist.barrier()
            _ev = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(_dc_N)]
            _loc = []
            for _s, _e in _ev:
                _dc_reset()
                torch.cuda.synchronize(); dist.barrier()
                _s.record()
                _pf6gm_mega(sp(), pf6_state)
                _e.record()
                torch.cuda.synchronize(); dist.barrier()
                _loc.append(_s.elapsed_time(_e) * 1e3)
            _aligned = _mok_rank_max(np.asarray(_loc, dtype=np.float64))
            return float(np.percentile(_aligned, 50)), float(np.percentile(_aligned, 95))

        # cumulative prefixes: phases 1..6 truncate; phase 0 = full run.
        _dc_phase_order = [1, 2, 3, 4, 5, 6, 0]
        _dc_cum = {}
        for _p in _dc_phase_order:
            _p50, _p95 = _dc_measure_phase(_p)
            _dc_cum[_p] = {"p50_us": _p50, "p95_us": _p95}
            if rank == 0:
                print(f"[PF6GM DECOMP] phase={_p} cum_p50={_p50:.1f}us cum_p95={_p95:.1f}us",
                      flush=True)
        # restore the canonical full-run desc so nothing downstream sees a truncated knob.
        pf6_state["desc_gm"] = _dc_orig_desc
        _dc_reset()
        torch.cuda.synchronize(); dist.barrier()

        # module marginals from cumulative prefixes (overlap-correct: later prefix contains
        # all earlier overlap). See design.md for the exact map.
        _c1 = _dc_cum[1]["p50_us"]; _c2 = _dc_cum[2]["p50_us"]; _c3 = _dc_cum[3]["p50_us"]
        _c4 = _dc_cum[4]["p50_us"]; _c5 = _dc_cum[5]["p50_us"]; _c6 = _dc_cum[6]["p50_us"]
        _c0 = _dc_cum[0]["p50_us"]
        _dc_modules = {
            "M0_retire_zero": _c1,
            "M1_qpush": _c2 - _c1,
            "M2_stream_verify": _c3 - _c2,
            "M3M4M5_plan_barriers_sort": _c4 - _c3,
            "M6_n2_phase1": _c5 - _c4,
            "M7_n2_phase2": _c6 - _c5,
            "M8M9_combine_arrival": _c0 - _c6,
        }
        # steady-state residual (skew + retire idle) = real same-run p50 minus the
        # barrier-synced epoch-1 full-run p50. NOT any single module's compute.
        _dc_steady = _mok_arms["pf6gm_mega"]["p50_us"]
        _dc_modules["steady_skew_retire_residual"] = _dc_steady - _c0
        R["pf6gm_decomp"] = {
            "warmup": _dc_W, "timed": _dc_N,
            "cumulative_p50_us": {str(k): v["p50_us"] for k, v in _dc_cum.items()},
            "cumulative_p95_us": {str(k): v["p95_us"] for k, v in _dc_cum.items()},
            "module_marginals_p50_us": _dc_modules,
            "steady_state_p50_us": _dc_steady,
            "barrier_synced_full_p50_us": _c0,
            "production_p50_us": _mok_prod["p50_us"],
            "note": (
                "reset-per-launch epoch-1 cumulative-prefix decomposition; module "
                "marginals are barrier-synced intrinsic costs; cross-rank skew + M0 "
                "retire-wait live in steady_skew_retire_residual, not in any module"
            ),
        }
        if rank == 0:
            print(f"[PF6GM DECOMP] modules_us={json.dumps(_dc_modules)}", flush=True)
            print(f"[PF6GM DECOMP] steady_p50={_dc_steady:.1f} full_synced_p50={_c0:.1f} "
                  f"prod_p50={_mok_prod['p50_us']:.1f}", flush=True)
    # ================= end exp_58 Phase 1 =================
    _mok_candidates = [name for name in R["arms"] if name != "production"]
    _mok_local_ok = bool(
        R["mok_correctness_all_pass"]
        and R["mok_control_fails"]
        and R.get("pf3_primitives", {}).get("pass_all_ranks", True)
        and all(
            arm["correctness"]["pass_all_ranks"] for arm in _mok_arms.values()
        )
    )
    _mok_all_ok = _all_ranks_pass(_mok_local_ok)
    _mok_flops = 6 * world * T * TOPK * H * INTER
    R["mok_eager"] = {
        "status": "valid_diagnostic",
        "reference_method": (
            "same-run production; Mixture-of-Kittens benchmarks/utils.py timing "
            "and tests/utils.py MXFP8 correctness policy"
        ),
        "graph_mode": False,
        "warmup_iters": MOK_WARMUP_ITERS,
        "timed_iters": MOK_TIMED_ITERS,
        "arm_order": list(R["arms"]),
        "timing": "HIP events, queued eager calls, index-aligned MAX(rank), median",
        "arms": _mok_arms,
        "ratios": {
            f"{name}/production_p50": _mok_arms[name]["p50_us"] / _mok_prod["p50_us"]
            for name in _mok_candidates
        } | {
            f"{name}/production_p95": _mok_arms[name]["p95_us"] / _mok_prod["p95_us"]
            for name in _mok_candidates
        },
        "routed_tflops_at_p50": {
            name: _mok_flops / (arm["p50_us"] * 1e6)
            for name, arm in _mok_arms.items()
        },
        "timing_gate_ok": _mok_all_ok,
        "scope": (
            "diagnostic synthetic eager benchmark; not a graph-mode production "
            "or promotion denominator"
        ),
    }
    R["timing_gate_ok"] = _mok_all_ok
    _finish_mok_benchmark(0 if _mok_all_ok else 2)

# ===================== graph capture every arm =====================
def capture(body):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): body(s.cuda_stream)       # 3 warmups (§12); monotonic gen/flags advance here too
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); hbarrier()
    with torch.cuda.graph(g, stream=s):
        body(torch.cuda.current_stream().cuda_stream)
    return g
graphs = {}; R["graph"] = {}
pf_poison_control_graph = None
pf2_poison_control_graph = None
k0d_noquant_control_graph = None
k0d_sq_b2f_wait_control_graph = None
cap_ok = True
try:
    for name, body, _ in ARMS:
        graphs[name] = capture(lambda st, b=body: b(st)); hbarrier()
    if "pf_full" in R["arms"]:
        pf_poison_control_graph = capture(pf_poison_control_body); hbarrier()
    if any(nm in R["arms"] for nm in ("pf2_full", "pf2_fused", "pf2_n2r")):
        pf2_poison_control_graph = capture(pf2_poison_control_body); hbarrier()
    if _K0D_SELECTED:
        k0d_noquant_control_graph = capture(k0d_noquant_downstream_control_body); hbarrier()
    if "k0d_sq_b2f" in _K0D_SELECTED:
        k0d_sq_b2f_wait_control_graph = capture(k0d_sq_b2f_wait_control_body); hbarrier()
except Exception as e:
    cap_ok = False
    R["graph_capture_err"] = "".join(traceback.format_exception(type(e), e, e.__traceback__)).splitlines()[-1]
if cap_ok and k0d_sq_b2f_wait_control_graph is not None:
    # The graph's three warmups plus capture advanced count by four epochs while the large flags
    # made those executions cheap. Zeroing only the flags now creates a persistent >=4-epoch gap.
    k0dsq_b2f_ctrl_flags.zero_(); torch.cuda.synchronize(); hbarrier()
R["graph_capture_ok"] = cap_ok

# Opt-in graph-resident phase profiler for the current decode push arms. Timing events are captured
# as graph nodes, so the intervals exclude Python launch gaps and retain the same kernel scheduling,
# cross-rank waits, and stream topology as the real regional graph. The extra event nodes make this
# diagnostic-only; the ordinary uninstrumented graphs remain the headline timing source.
if (
    cap_ok
    and int(os.environ.get("K0_DECODE_GRAPH_EVENT_PROFILE", "0"))
    and T <= 512
    and _DECODE_PHASE_ARMS
):
    _decode_phase_reps = int(os.environ.get("K0_DECODE_PHASE_REPS", "100"))
    _decode_phase_warmups = int(os.environ.get("K0_DECODE_PHASE_WARMUPS", "10"))
    _decode_graph_stage_names = {
        "production": ("dispatch", "aiter_gemm", "combine"),
        "k0d_pd": ("quant", "push", "dsort", "n2_p1", "n2_p2", "combine_edge_c"),
        "k0d_mega": ("quant", "mega_ingress", "n2_p1", "n2_p2", "combine_edge_c"),
    }
    _decode_profile_graphs = {}
    _decode_profile_events = {}
    _decode_profile_holds = {}

    def _decode_profile_body(_nm, _events, _hold):
        _st = torch.cuda.current_stream().cuda_stream
        _events[0].record()
        if _nm == "production":
            d_a1, d_wgt, d_scale, d_ids, recv = op_prod.dispatch(
                hidden, topk_wgt, None, topk_ids, BLK, -1, WARP,
            )
            _events[1].record()
            _fused = fused_moe(
                d_a1[:M_TRIM], w1_b, w2_b, d_wgt[:M_TRIM], d_ids[:M_TRIM],
                expert_mask=expert_mask, activation=ActivationType.Silu, quant_type=QT,
                w1_scale=fc1_scale, w2_scale=fc2_scale, a1_scale=None,
                num_local_tokens=recv, doweight_stage1=False, dtype=torch.bfloat16,
            )
            _events[2].record()
            _res = op_prod.combine(_fused, None, topk_ids, BLK, -1, WARP)[0][:T]
            _events[3].record()
            _hold["dispatch"] = (d_a1, d_wgt, d_scale, d_ids, recv)
            _hold["fused"] = _fused
            _hold["result"] = _res
        elif _nm == "k0d_pd":
            _quant(_st); _events[1].record()
            _k0d_push_ingress(_st); _events[2].record()
            _k0d_dsort_ingress(_st); _events[3].record()
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _events[4].record()
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _events[5].record()
            _k0d_combine_pd(_st); _events[6].record()
        else:
            _quant(_st); _events[1].record()
            _k0d_mega_ingress(_st); _events[2].record()
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _events[3].record()
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _events[4].record()
            _k0d_combine_pd(_st); _events[5].record()

    try:
        for _nm in _DECODE_PHASE_ARMS:
            _names = _decode_graph_stage_names[_nm]
            _events = [torch.cuda.Event(True) for _ in range(len(_names) + 1)]
            _hold = {}
            _g = torch.cuda.CUDAGraph()
            _s = torch.cuda.Stream()
            with torch.cuda.stream(_s):
                for _ in range(3):
                    _decode_profile_body(_nm, _events, _hold)
            torch.cuda.current_stream().wait_stream(_s)
            torch.cuda.synchronize()
            hbarrier()
            with torch.cuda.graph(_g, stream=_s):
                _decode_profile_body(_nm, _events, _hold)
            _decode_profile_graphs[_nm] = _g
            _decode_profile_events[_nm] = _events
            _decode_profile_holds[_nm] = _hold
            hbarrier()
    except Exception as _decode_graph_profile_error:
        raise RuntimeError(
            "decode graph phase capture failed: "
            + "".join(
                traceback.format_exception_only(
                    type(_decode_graph_profile_error), _decode_graph_profile_error
                )
            ).strip()
        )

    # Capture the pure owner-pull body separately, with all edge-C rendezvous pointers null.
    _plain_events = [torch.cuda.Event(True), torch.cuda.Event(True)]
    _plain_graph = torch.cuda.CUDAGraph()
    _plain_stream = torch.cuda.Stream()
    with torch.cuda.stream(_plain_stream):
        for _ in range(3):
            _plain_events[0].record()
            _k0d_combine(_plain_stream.cuda_stream, folded=False)
            _plain_events[1].record()
    torch.cuda.current_stream().wait_stream(_plain_stream)
    torch.cuda.synchronize()
    hbarrier()
    with torch.cuda.graph(_plain_graph, stream=_plain_stream):
        _plain_events[0].record()
        _k0d_combine(torch.cuda.current_stream().cuda_stream, folded=False)
        _plain_events[1].record()
    hbarrier()

    # Warm every captured profile graph in the same cyclic order used for sampling.
    for _rep in range(_decode_phase_warmups):
        _rotation = _rep % len(_DECODE_PHASE_ARMS)
        _order = _DECODE_PHASE_ARMS[_rotation:] + _DECODE_PHASE_ARMS[:_rotation]
        for _nm in _order:
            hbarrier()
            _decode_profile_graphs[_nm].replay()
            torch.cuda.synchronize()
            hbarrier()
    for _ in range(_decode_phase_warmups):
        hbarrier(); _plain_graph.replay(); torch.cuda.synchronize(); hbarrier()

    _decode_graph_samples = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_graph_cumulative = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_graph_pperr = {nm: [] for nm in _DECODE_PHASE_ARMS if nm != "production"}
    _decode_graph_orders = []
    for _rep in range(_decode_phase_reps):
        _rotation = _rep % len(_DECODE_PHASE_ARMS)
        _order = _DECODE_PHASE_ARMS[_rotation:] + _DECODE_PHASE_ARMS[:_rotation]
        _decode_graph_orders.append(_order)
        for _nm in _order:
            hbarrier()
            if _nm != "production":
                pperr.zero_(); torch.cuda.synchronize()
            _decode_profile_graphs[_nm].replay()
            torch.cuda.synchronize()
            _events = _decode_profile_events[_nm]
            _cum = [float(_events[0].elapsed_time(e) * 1e3) for e in _events[1:]]
            _dur = [_cum[0]] + [
                float(_cum[i] - _cum[i - 1]) for i in range(1, len(_cum))
            ]
            _decode_graph_samples[_nm].append(_dur)
            _decode_graph_cumulative[_nm].append(_cum)
            if _nm != "production":
                _decode_graph_pperr[_nm].append(int(pperr.item()))
            hbarrier()

    _plain_graph_samples = []
    for _rep in range(_decode_phase_reps):
        hbarrier()
        _plain_graph.replay()
        torch.cuda.synchronize()
        _plain_graph_samples.append(
            float(_plain_events[0].elapsed_time(_plain_events[1]) * 1e3)
        )
        hbarrier()

    _decode_graph_profile = dict(
        reps=_decode_phase_reps,
        warmups=_decode_phase_warmups,
        units="microseconds",
        graph_mode=True,
        diagnostic_event_nodes=True,
        order_policy="cyclic rotation per replay",
        orders=_decode_graph_orders,
        plain_owner_pull_combine_us=_plain_graph_samples,
        arms={},
    )
    _decode_graph_local_ok = True
    for _nm in _DECODE_PHASE_ARMS:
        _arr = np.asarray(_decode_graph_samples[_nm], dtype=np.float64)
        _decode_graph_profile["arms"][_nm] = dict(
            stage_names=list(_decode_graph_stage_names[_nm]),
            duration_us=_decode_graph_samples[_nm],
            cumulative_us=_decode_graph_cumulative[_nm],
            local_p50_us={
                n: float(np.median(_arr[:, i]))
                for i, n in enumerate(_decode_graph_stage_names[_nm])
            },
            local_p95_us={
                n: float(np.percentile(_arr[:, i], 95))
                for i, n in enumerate(_decode_graph_stage_names[_nm])
            },
            pperr=_decode_graph_pperr.get(_nm, []),
        )
        if _nm != "production":
            _decode_graph_local_ok = bool(
                _decode_graph_local_ok and all(x == 0 for x in _decode_graph_pperr[_nm])
            )
    _decode_graph_profile["local_protocol_ok"] = _decode_graph_local_ok
    _decode_graph_profile["pass_all_ranks"] = _all_ranks_pass(_decode_graph_local_ok)
    R["decode_phase_profile"] = _decode_graph_profile
    if rank == 0:
        print(
            f"[K0D GRAPH PHASE PROFILE] reps={_decode_phase_reps} "
            f"arms={_DECODE_PHASE_ARMS} local_p50="
            f"{ {nm: _decode_graph_profile['arms'][nm]['local_p50_us'] for nm in _DECODE_PHASE_ARMS} } "
            f"plain_combine_p50={float(np.median(_plain_graph_samples))}",
            flush=True,
        )
    if not _decode_graph_profile["pass_all_ranks"]:
        raise RuntimeError("decode graph phase profile observed a protocol error")
    pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# Graph-replay-safe decode attribution. Captured torch events on this ROCm stack retain the
# timestamp from capture, so the event-node diagnostic above is kept behind a separate legacy
# switch. These marker kernels write s_memrealtime on every replay. An empty marker-to-marker
# graph measures the extra node cost, and a short timer-spin kernel calibrates ticks to microseconds.
if (
    cap_ok
    and int(os.environ.get("K0_DECODE_PHASE_PROFILE", "0"))
    and (T <= 512 or _PF3_SELECTED)
    and _DECODE_PHASE_ARMS
):
    _decode_phase_reps = int(os.environ.get("K0_DECODE_PHASE_REPS", "100"))
    _decode_phase_warmups = int(os.environ.get("K0_DECODE_PHASE_WARMUPS", "10"))
    _decode_marker_stage_names = {
        "production": ("dispatch", "aiter_gemm", "combine"),
        "k0d_pd": ("quant", "push", "dsort", "n2_p1", "n2_p2", "combine_edge_c"),
        "k0d_mega": ("quant", "mega_ingress", "n2_p1", "n2_p2", "combine_edge_c"),
        "pf3_pd": ("qpush", "dsort", "n2_p1", "n2_p2", "barrier2", "combine"),
        "pf4h": ("qpush", "dsort", "n2_p1", "n2_p2", "barrier2", "combine"),
        "pf3_mega": ("mega", "n2_p1", "n2_p2", "barrier2", "combine"),
    }

    def _decode_mark(_stream, _stamps, _index):
        fn_k0d_phase_marker.launch(
            (1,), (1,), 0, _stream, _stamps.data_ptr(), int(_index)
        )

    def _decode_marker_body(_nm, _stamps, _hold):
        _st = torch.cuda.current_stream().cuda_stream
        _decode_mark(_st, _stamps, 0)
        if _nm == "production":
            d_a1, d_wgt, d_scale, d_ids, recv = op_prod.dispatch(
                hidden, topk_wgt, None, topk_ids, BLK, -1, WARP,
            )
            _decode_mark(_st, _stamps, 1)
            _fused = fused_moe(
                d_a1[:M_TRIM], w1_b, w2_b, d_wgt[:M_TRIM], d_ids[:M_TRIM],
                expert_mask=expert_mask, activation=ActivationType.Silu, quant_type=QT,
                w1_scale=fc1_scale, w2_scale=fc2_scale, a1_scale=None,
                num_local_tokens=recv, doweight_stage1=False, dtype=torch.bfloat16,
            )
            _decode_mark(_st, _stamps, 2)
            _res = op_prod.combine(_fused, None, topk_ids, BLK, -1, WARP)[0][:T]
            _decode_mark(_st, _stamps, 3)
            _hold["dispatch"] = (d_a1, d_wgt, d_scale, d_ids, recv)
            _hold["fused"] = _fused
            _hold["result"] = _res
        elif _nm == "k0d_pd":
            _quant(_st)
            _decode_mark(_st, _stamps, 1)
            _k0d_push_ingress(_st)
            _decode_mark(_st, _stamps, 2)
            _k0d_dsort_ingress(_st)
            _decode_mark(_st, _stamps, 3)
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _decode_mark(_st, _stamps, 4)
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _decode_mark(_st, _stamps, 5)
            _k0d_combine_pd(_st)
            _decode_mark(_st, _stamps, 6)
        elif _nm == "k0d_mega":
            _quant(_st)
            _decode_mark(_st, _stamps, 1)
            _k0d_mega_ingress(_st)
            _decode_mark(_st, _stamps, 2)
            k0_n2.n2_phase1(
                a_dst_pd, sc_dst, w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _decode_mark(_st, _stamps, 3)
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _decode_mark(_st, _stamps, 4)
            _k0d_combine_pd(_st)
            _decode_mark(_st, _stamps, 5)
        elif _nm in ("pf3_pd", "pf4h"):
            _st3 = pf3_state[_nm]
            _pf3_qpush(_st, _st3)
            _decode_mark(_st, _stamps, 1)
            if _nm == "pf4h":
                _pf4_dsort(_st, _st3)
            else:
                _pf3_dsort(_st, _st3)
            _decode_mark(_st, _stamps, 2)
            k0_n2.n2_phase1(
                _st3["a_dst"], _st3["sc_dst"], w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _decode_mark(_st, _stamps, 3)
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _decode_mark(_st, _stamps, 4)
            ms.shmem_barrier_on_stream(_st)
            _decode_mark(_st, _stamps, 5)
            fn_combine_pf.launch(
                (1024,), (256,), 0, _st, cand_out.data_ptr(), part_p,
                pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(), T, H, rank,
            )
            _decode_mark(_st, _stamps, 6)
        else:
            _st3 = pf3_state["pf3_mega"]
            _pf3_mega(_st, _st3)
            _decode_mark(_st, _stamps, 1)
            k0_n2.n2_phase1(
                _st3["a_dst"], _st3["sc_dst"], w13, fc1_scale, sti1, sei1, nvi1,
                _n2_bufs["a2q"], _n2_bufs["dq2"], _st,
            )
            _decode_mark(_st, _stamps, 2)
            k0_n2.n2_phase2(
                _n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale,
                sti1, swt1, sei1, nvi1, part, _st,
            )
            _decode_mark(_st, _stamps, 3)
            ms.shmem_barrier_on_stream(_st)
            _decode_mark(_st, _stamps, 4)
            fn_combine_pf.launch(
                (1024,), (256,), 0, _st, cand_out.data_ptr(), part_p,
                pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(), T, H, rank,
            )
            _decode_mark(_st, _stamps, 5)

    # Calibrate s_memrealtime instead of assuming its nominal 100 MHz period.
    _timer_calibration_stamps = torch.zeros((2,), dtype=torch.int64, device="cuda")
    _timer_calibration_tick_us = []
    _timer_calibration_event_us = []
    _timer_target_ticks = 1000000
    for _ in range(5):
        _cal_begin = torch.cuda.Event(True)
        _cal_end = torch.cuda.Event(True)
        _cal_begin.record()
        fn_k0d_phase_timer_calibration.launch(
            (1,), (1,), 0, sp(), _timer_calibration_stamps.data_ptr(), _timer_target_ticks
        )
        _cal_end.record()
        torch.cuda.synchronize()
        _cal_ticks = int(
            _timer_calibration_stamps[1].item() - _timer_calibration_stamps[0].item()
        )
        _cal_us = float(_cal_begin.elapsed_time(_cal_end) * 1e3)
        if _cal_ticks <= 0:
            raise RuntimeError(f"invalid s_memrealtime calibration delta: {_cal_ticks}")
        _timer_calibration_tick_us.append(_cal_us / _cal_ticks)
        _timer_calibration_event_us.append(_cal_us)
    _timer_tick_us = float(np.median(_timer_calibration_tick_us))
    hbarrier()

    _decode_marker_graphs = {}
    _decode_marker_stamps = {}
    _decode_marker_holds = {}
    try:
        for _nm in _DECODE_PHASE_ARMS:
            _nstamps = len(_decode_marker_stage_names[_nm]) + 1
            _stamps = torch.zeros((_nstamps,), dtype=torch.int64, device="cuda")
            _hold = {}
            _graph = torch.cuda.CUDAGraph()
            _stream = torch.cuda.Stream()
            with torch.cuda.stream(_stream):
                for _ in range(3):
                    _decode_marker_body(_nm, _stamps, _hold)
            torch.cuda.current_stream().wait_stream(_stream)
            torch.cuda.synchronize()
            hbarrier()
            with torch.cuda.graph(_graph, stream=_stream):
                _decode_marker_body(_nm, _stamps, _hold)
            _decode_marker_graphs[_nm] = _graph
            _decode_marker_stamps[_nm] = _stamps
            _decode_marker_holds[_nm] = _hold
            hbarrier()
    except Exception as _decode_marker_capture_error:
        raise RuntimeError(
            "decode device-marker graph capture failed: "
            + "".join(
                traceback.format_exception_only(
                    type(_decode_marker_capture_error), _decode_marker_capture_error
                )
            ).strip()
        )

    # The plain-combine calibration bracket is regime-dependent: decode uses k0d_combine,
    # prefill uses k0pf_combine (the only combine loaded in a pf3 campaign).
    if K0D_REQUESTED:
        def _plain_combine(_st):
            _k0d_combine(_st, folded=False)
    else:
        def _plain_combine(_st):
            fn_combine_pf.launch(
                (1024,), (256,), 0, _st, cand_out.data_ptr(), part_p,
                pull_ptr.data_ptr(), pull_src.data_ptr(), stage.data_ptr(), T, H, rank,
            )
    _plain_marker_stamps = torch.zeros((2,), dtype=torch.int64, device="cuda")
    _plain_marker_graph = torch.cuda.CUDAGraph()
    _plain_marker_stream = torch.cuda.Stream()
    with torch.cuda.stream(_plain_marker_stream):
        for _ in range(3):
            _decode_mark(_plain_marker_stream.cuda_stream, _plain_marker_stamps, 0)
            _plain_combine(_plain_marker_stream.cuda_stream)
            _decode_mark(_plain_marker_stream.cuda_stream, _plain_marker_stamps, 1)
    torch.cuda.current_stream().wait_stream(_plain_marker_stream)
    torch.cuda.synchronize()
    hbarrier()
    with torch.cuda.graph(_plain_marker_graph, stream=_plain_marker_stream):
        _plain_st = torch.cuda.current_stream().cuda_stream
        _decode_mark(_plain_st, _plain_marker_stamps, 0)
        _plain_combine(_plain_st)
        _decode_mark(_plain_st, _plain_marker_stamps, 1)
    hbarrier()

    _empty_marker_stamps = torch.zeros((2,), dtype=torch.int64, device="cuda")
    _empty_marker_graph = torch.cuda.CUDAGraph()
    _empty_marker_stream = torch.cuda.Stream()
    with torch.cuda.stream(_empty_marker_stream):
        for _ in range(3):
            _decode_mark(_empty_marker_stream.cuda_stream, _empty_marker_stamps, 0)
            _decode_mark(_empty_marker_stream.cuda_stream, _empty_marker_stamps, 1)
    torch.cuda.current_stream().wait_stream(_empty_marker_stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(_empty_marker_graph, stream=_empty_marker_stream):
        _empty_st = torch.cuda.current_stream().cuda_stream
        _decode_mark(_empty_st, _empty_marker_stamps, 0)
        _decode_mark(_empty_st, _empty_marker_stamps, 1)
    hbarrier()

    for _rep in range(_decode_phase_warmups):
        _rotation = _rep % len(_DECODE_PHASE_ARMS)
        _order = _DECODE_PHASE_ARMS[_rotation:] + _DECODE_PHASE_ARMS[:_rotation]
        for _nm in _order:
            hbarrier()
            _decode_marker_graphs[_nm].replay()
            torch.cuda.synchronize()
            hbarrier()
    for _ in range(_decode_phase_warmups):
        hbarrier()
        _plain_marker_graph.replay()
        torch.cuda.synchronize()
        hbarrier()
        _empty_marker_graph.replay()
        torch.cuda.synchronize()
        hbarrier()

    _decode_marker_ticks = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_marker_outer_us = {nm: [] for nm in _DECODE_PHASE_ARMS}
    _decode_marker_pperr = {nm: [] for nm in _DECODE_PHASE_ARMS if nm != "production"}
    _decode_marker_orders = []
    for _rep in range(_decode_phase_reps):
        _rotation = _rep % len(_DECODE_PHASE_ARMS)
        _order = _DECODE_PHASE_ARMS[_rotation:] + _DECODE_PHASE_ARMS[:_rotation]
        _decode_marker_orders.append(_order)
        for _nm in _order:
            if _nm != "production":
                pperr.zero_()
                torch.cuda.synchronize()
            hbarrier()
            _outer_begin = torch.cuda.Event(True)
            _outer_end = torch.cuda.Event(True)
            _outer_begin.record()
            _decode_marker_graphs[_nm].replay()
            _outer_end.record()
            torch.cuda.synchronize()
            _ticks = [int(x) for x in _decode_marker_stamps[_nm].cpu().tolist()]
            if any(_ticks[i + 1] <= _ticks[i] for i in range(len(_ticks) - 1)):
                raise RuntimeError(f"non-monotonic device marker timestamps for {_nm}: {_ticks}")
            _decode_marker_ticks[_nm].append(_ticks)
            _decode_marker_outer_us[_nm].append(
                float(_outer_begin.elapsed_time(_outer_end) * 1e3)
            )
            if _nm != "production":
                _decode_marker_pperr[_nm].append(int(pperr.item()))
            hbarrier()

    _plain_marker_ticks = []
    _plain_marker_outer_us = []
    _empty_marker_ticks = []
    _empty_marker_outer_us = []
    for _rep in range(_decode_phase_reps):
        hbarrier()
        _outer_begin = torch.cuda.Event(True)
        _outer_end = torch.cuda.Event(True)
        _outer_begin.record()
        _plain_marker_graph.replay()
        _outer_end.record()
        torch.cuda.synchronize()
        _plain_ticks = [int(x) for x in _plain_marker_stamps.cpu().tolist()]
        _plain_marker_ticks.append(_plain_ticks)
        _plain_marker_outer_us.append(float(_outer_begin.elapsed_time(_outer_end) * 1e3))
        hbarrier()
        _outer_begin = torch.cuda.Event(True)
        _outer_end = torch.cuda.Event(True)
        _outer_begin.record()
        _empty_marker_graph.replay()
        _outer_end.record()
        torch.cuda.synchronize()
        _empty_ticks = [int(x) for x in _empty_marker_stamps.cpu().tolist()]
        _empty_marker_ticks.append(_empty_ticks)
        _empty_marker_outer_us.append(float(_outer_begin.elapsed_time(_outer_end) * 1e3))
        hbarrier()

    _empty_delta_ticks = np.diff(
        np.asarray(_empty_marker_ticks, dtype=np.int64), axis=1
    ).reshape(-1)
    _marker_interval_overhead_ticks = float(np.median(_empty_delta_ticks))
    _marker_interval_overhead_us = _marker_interval_overhead_ticks * _timer_tick_us
    _decode_marker_profile = dict(
        reps=_decode_phase_reps,
        warmups=_decode_phase_warmups,
        units="microseconds",
        graph_mode=True,
        fresh_device_timestamps_each_replay=True,
        timer="s_memrealtime",
        timer_target_ticks=_timer_target_ticks,
        timer_calibration_tick_us=_timer_calibration_tick_us,
        timer_calibration_event_us=_timer_calibration_event_us,
        timer_tick_us=_timer_tick_us,
        empty_marker_ticks=_empty_marker_ticks,
        empty_marker_outer_us=_empty_marker_outer_us,
        marker_interval_overhead_ticks=_marker_interval_overhead_ticks,
        marker_interval_overhead_us=_marker_interval_overhead_us,
        order_policy="cyclic rotation per replay",
        orders=_decode_marker_orders,
        plain_owner_pull_combine=dict(
            boundary_ticks=_plain_marker_ticks,
            outer_event_us=_plain_marker_outer_us,
        ),
        arms={},
    )
    _decode_marker_local_ok = True
    for _nm in _DECODE_PHASE_ARMS:
        _boundary = np.asarray(_decode_marker_ticks[_nm], dtype=np.int64)
        _raw_duration_us = np.diff(_boundary, axis=1).astype(np.float64) * _timer_tick_us
        _corrected_duration_us = np.maximum(
            0.0, _raw_duration_us - _marker_interval_overhead_us
        )
        _decode_marker_profile["arms"][_nm] = dict(
            stage_names=list(_decode_marker_stage_names[_nm]),
            boundary_ticks=_decode_marker_ticks[_nm],
            outer_event_us=_decode_marker_outer_us[_nm],
            raw_duration_us=_raw_duration_us.tolist(),
            corrected_duration_us=_corrected_duration_us.tolist(),
            corrected_local_p50_us={
                n: float(np.median(_corrected_duration_us[:, i]))
                for i, n in enumerate(_decode_marker_stage_names[_nm])
            },
            corrected_local_p95_us={
                n: float(np.percentile(_corrected_duration_us[:, i], 95))
                for i, n in enumerate(_decode_marker_stage_names[_nm])
            },
            pperr=_decode_marker_pperr.get(_nm, []),
        )
        if _nm != "production":
            _decode_marker_local_ok = bool(
                _decode_marker_local_ok
                and all(x == 0 for x in _decode_marker_pperr[_nm])
            )
    _decode_marker_profile["local_protocol_ok"] = _decode_marker_local_ok
    _decode_marker_profile["pass_all_ranks"] = _all_ranks_pass(_decode_marker_local_ok)
    R["decode_device_phase_profile"] = _decode_marker_profile
    if rank == 0:
        print(
            f"[K0D DEVICE PHASE PROFILE] reps={_decode_phase_reps} "
            f"tick_us={_timer_tick_us:.9f} marker_us={_marker_interval_overhead_us:.3f} "
            f"local_p50="
            f"{ {nm: _decode_marker_profile['arms'][nm]['corrected_local_p50_us'] for nm in _DECODE_PHASE_ARMS} }",
            flush=True,
        )
    if not _decode_marker_profile["pass_all_ranks"]:
        raise RuntimeError("decode device phase profile observed a protocol error")
    pperr.zero_()
    torch.cuda.synchronize()
    hbarrier()

# ===================== graph correctness gate on every arm (THE gate that makes the number real) =====================
# candidate output vs FRESH same-run stock, frozen tolerance, never widened. This + control_fails +
# pperr==0 is the timing gate. Repeated and interleaved graph stress below are also hard gates.
gate_ok = bool(
    cap_ok
    and R["eager_all_pass"]
    and R.get("pf3_primitives", {}).get("pass_all_ranks", True)
)
if cap_ok:
    for name, _, buf in ARMS:
        if name not in _PROD_LIKE: cand_out.zero_()
        perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        graphs[name].replay(); torch.cuda.synchronize()
        R["graph"][name] = gate(_obuf(name)); R["graph"][name]["pass"] = ok_arm(name, R["graph"][name])
        R["graph"][name]["pperr"] = int(pperr.cpu().numpy().reshape(-1)[0]); hbarrier()
        R["graph"][name]["plan_err"] = int(perr.cpu().numpy().reshape(-1)[0]) if name in _CUSTOM_PLAN_ARMS else 0
        _graph_local = bool(
            R["graph"][name]["pass"]
            and R["graph"][name]["plan_err"] == 0
            and R["graph"][name]["pperr"] == 0
        )
        R["graph"][name]["local_pass"] = _graph_local
        R["graph"][name]["pass"] = _all_ranks_pass(_graph_local)
        if not R["graph"][name]["pass"]: gate_ok = False
        if rank == 0: print(f"[MARK] graph {name} rel_L2={R['graph'][name]['rel_L2']:.5f} pass={R['graph'][name]['pass']} pperr={R['graph'][name]['pperr']}", flush=True)
    gate_ok = gate_ok and R["control_fails"]
R["timing_gate_ok"] = bool(gate_ok)

# PF3 reconstructs every receive row, scale, plan invariant, partial row, and combine CSR on each
# replay. Poison all of those sinks and require 16 consecutive captured replays to recover.
R["k0pf3_epoch_ok"] = True
if cap_ok and _PF3_SELECTED:
    _pf3_epoch_arms = {}
    _pf3_epoch_local = True
    for _pf3_nm in _PF3_SELECTED:
        _st3 = pf3_state[_pf3_nm]
        _pf3_rows = []
        _pf3_arm_ok = True
        for _rep3 in range(16):
            _st3["a_dst_u8"].fill_(0x7E)
            _st3["sc_stage"].fill_(float("nan"))
            _st3["recv_eid"].fill_(-1)
            _st3["recv_wgt"].fill_(float("nan"))
            _st3["pull_stage"].fill_(-1)
            _st3["pull_cnt"].fill_(-1)
            nvi.zero_(); pull_ptr.zero_(); pull_src.fill_(-1)
            part[:T_loc].fill_(1000.0)
            cand_out.zero_(); pperr.zero_()
            torch.cuda.synchronize(); hbarrier()
            graphs[_pf3_nm].replay(); torch.cuda.synchronize()
            _g3 = gate(cand_out)
            _e3 = int(pperr.item())
            _n3 = [int(x) for x in nvi.reshape(-1).cpu().numpy()]
            _p3 = int(pull_ptr[T].item())
            _ok3 = bool(
                ok(_g3) and _e3 == 0 and _n3 == [_padded, T_loc] and _p3 == _pull_n
            )
            _pf3_arm_ok = bool(_pf3_arm_ok and _ok3)
            _pf3_rows.append(dict(
                rep=_rep3, pass_replay=_ok3, rel_L2=_g3["rel_L2"],
                max_abs=_g3["max_abs"], pperr=_e3, nvi=_n3, pull_n=_p3,
            ))
            hbarrier()
        _pf3_epoch_local = bool(_pf3_epoch_local and _pf3_arm_ok)
        _pf3_epoch_arms[_pf3_nm] = dict(candidate_16_pass=_pf3_arm_ok, replays=_pf3_rows)
    _pf3_epoch_all = _all_ranks_pass(_pf3_epoch_local)
    R["k0pf3_epoch"] = dict(
        pass_all_ranks=_pf3_epoch_all, local_pass=_pf3_epoch_local,
        arms=_pf3_epoch_arms,
    )
    R["k0pf3_epoch_ok"] = _pf3_epoch_all
    gate_ok = bool(gate_ok and _pf3_epoch_all)
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(
            f"[K0PF3 GATE] epoch poison pass={_pf3_epoch_all} "
            f"arms={_PF3_SELECTED}",
            flush=True,
        )
    pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# The corpus route is static, so normal graph correctness cannot prove that the pybind plan launch
# was actually captured. Poison its decisive device outputs and require each custom graph to rebuild
# them during replay. This catches a default-stream launch silently falling outside capture.
R["plan_capture"] = {}
if cap_ok:
    for _pnm in ("frozen_n2", "frozen_n2r", "pf_full", "pf2_plan", "pf2_push", "pf2_full", "pf2_fused", "pf2_n2r", "k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f"):
        if _pnm not in R["arms"]:
            continue
        nvi.zero_(); pull_ptr.zero_(); perr.zero_(); pperr.zero_(); cand_out.zero_(); torch.cuda.synchronize(); hbarrier()
        graphs[_pnm].replay(); torch.cuda.synchronize()
        _pcg = gate(cand_out)
        _pcerr = int(perr.cpu().numpy().reshape(-1)[0])
        _pcperr = int(pperr.cpu().numpy().reshape(-1)[0])
        _pcnvi = [int(x) for x in nvi.cpu().numpy().reshape(-1)]
        _pcok = bool(ok(_pcg) and _pcerr == 0 and _pcperr == 0 and _pcnvi == [_padded, T_loc])
        _pcall = _all_ranks_pass(_pcok)
        R["plan_capture"][_pnm] = dict(pass_all_ranks=_pcall, local_pass=_pcok,
                                       rel_L2=_pcg["rel_L2"], plan_err=_pcerr, pperr=_pcperr, nvi=_pcnvi)
        gate_ok = bool(gate_ok and _pcall)
        if rank == 0:
            print(f"[K0PF GATE] plan_capture {_pnm} pass={_pcall} "
                  f"nvi={_pcnvi} rel_L2={_pcg['rel_L2']:.5f}", flush=True)
        hbarrier()
R["timing_gate_ok"] = bool(gate_ok)

# ===================== k0pf epoch-positive poison gate =====================
R["k0pf_epoch_ok"] = True
if cap_ok and "pf_full" in R["arms"] and pf_poison_control_graph is not None:
    _candidate_ok = True
    _control_failed = True
    _epoch_rows = []
    for rep in range(16):
        a_src_u8.fill_(0x7E); sc_src.fill_(float("nan"))
        cand_out.zero_(); torch.cuda.synchronize(); hbarrier()
        pf_poison_control_graph.replay(); torch.cuda.synchronize()
        _cg = gate(cand_out)
        _cfail = bool(not ok(_cg))
        _control_failed = bool(_control_failed and _cfail)
        hbarrier()
        cand_out.zero_(); torch.cuda.synchronize()
        graphs["pf_full"].replay(); torch.cuda.synchronize()
        _pg = gate(cand_out)
        _pe = int(perr.cpu().numpy().reshape(-1)[0])
        _ppass = bool(ok(_pg) and _pe == 0)
        _candidate_ok = bool(_candidate_ok and _ppass)
        _epoch_rows.append(dict(rep=rep, control_rel_L2=_cg["rel_L2"],
                                control_failed=_cfail, candidate_rel_L2=_pg["rel_L2"],
                                candidate_pass=_ppass, plan_err=_pe))
        hbarrier()
    _epoch_local = bool(_candidate_ok and _control_failed)
    _epoch_all = _all_ranks_pass(_epoch_local)
    R["k0pf_epoch"] = dict(pass_all_ranks=_epoch_all, local_pass=_epoch_local,
                            candidate_16_pass=_candidate_ok,
                            poison_control_16_fail=_control_failed, replays=_epoch_rows)
    R["k0pf_epoch_ok"] = _epoch_all
    gate_ok = bool(gate_ok and _epoch_all)
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(f"[K0PF GATE] epoch poison pass={_epoch_all} "
              f"candidate_16_pass={_candidate_ok} control_16_fail={_control_failed}", flush=True)
    hbarrier()

# Source-push epoch gate. The no-quant control must deliver poisoned bytes and fail; every selected
# full k0pf2 graph must rebuild the source, publish the current epoch, and pass 16/16.
R["k0pf2_epoch_ok"] = True
_pf2_epoch_arms = [nm for nm in ("pf2_full", "pf2_fused", "pf2_n2r") if nm in R["arms"]]
if cap_ok and _pf2_epoch_arms and pf2_poison_control_graph is not None:
    _pf2_control_failed = True
    _pf2_epoch_rows = {}
    _pf2_all_candidates = True
    for _p2nm in _pf2_epoch_arms:
        _rows = []
        _candidate_ok = True
        for rep in range(16):
            a_src_u8.fill_(0x7E); sc_src.fill_(float("nan"))
            cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
            pf2_poison_control_graph.replay(); torch.cuda.synchronize()
            _cg = gate(cand_out)
            _cfail = bool(not ok(_cg))
            _pf2_control_failed = bool(_pf2_control_failed and _cfail)
            hbarrier()
            cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize()
            graphs[_p2nm].replay(); torch.cuda.synchronize()
            _pg = gate(cand_out)
            _pe = int(perr.cpu().numpy().reshape(-1)[0])
            _ppe = int(pperr.cpu().numpy().reshape(-1)[0])
            _ppass = bool(ok(_pg) and _pe == 0 and _ppe == 0)
            _candidate_ok = bool(_candidate_ok and _ppass)
            _rows.append(
                dict(
                    rep=rep,
                    control_rel_L2=_cg["rel_L2"],
                    control_failed=_cfail,
                    candidate_rel_L2=_pg["rel_L2"],
                    candidate_pass=_ppass,
                    plan_err=_pe,
                    push_err=_ppe,
                )
            )
            hbarrier()
        _pf2_all_candidates = bool(_pf2_all_candidates and _candidate_ok)
        _pf2_epoch_rows[_p2nm] = dict(candidate_16_pass=_candidate_ok, replays=_rows)
    _pf2_epoch_local = bool(_pf2_all_candidates and _pf2_control_failed)
    _pf2_epoch_all = _all_ranks_pass(_pf2_epoch_local)
    R["k0pf2_epoch"] = dict(
        pass_all_ranks=_pf2_epoch_all,
        local_pass=_pf2_epoch_local,
        poison_control_fails=_pf2_control_failed,
        arms=_pf2_epoch_rows,
    )
    R["k0pf2_epoch_ok"] = _pf2_epoch_all
    gate_ok = bool(gate_ok and _pf2_epoch_all)
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(
            f"[K0PF2 GATE] epoch poison pass={_pf2_epoch_all} "
            f"arms={_pf2_epoch_arms} control_fails={_pf2_control_failed}",
            flush=True,
        )
    hbarrier()

# ===================== k0d 16-replay poison / plan / partial epoch gate =====================
# Poison all cross-rank ingress payload, decisive plan outputs, and the live symmetric partial sink.
# The captured k0d graph must reconstruct every one.  Its paired no-quant downstream graph uses
# k0d gather+zero/n2/k0d combine but deliberately omits origin quant, so it must fail every epoch.
R["k0d_epoch_ok"] = True
if cap_ok and _K0D_SELECTED:
    _k0d_epoch_rows = {}
    _k0d_epoch_all_local = bool(k0d_noquant_control_graph is not None)
    _k0d_control_all_failed = True

    def _k0df_state_snapshot():
        return dict(
            quant_count=int(k0df_quant_count.item()), arr_count=int(k0df_arr_count.item()),
            don_count=int(k0df_don_count.item()),
            quant_flags=[int(x) for x in k0df_quant_flags.reshape(-1).cpu().numpy()],
            comb_flags=[int(x) for x in k0df_comb_flags.reshape(-1).cpu().numpy()],
            arr_flags=[int(x) for x in k0df_arr_flags.reshape(-1).cpu().numpy()],
        )

    def _k0dsq_b2f_state_snapshot():
        return dict(
            b1_count=int(k0ds_b1_count.item()),
            arr_count=int(k0dsq_b2f_arr_count.item()),
            b1_flags=[int(x) for x in k0ds_b1_flags.reshape(-1).cpu().numpy()],
            arr_flags=[int(x) for x in k0dsq_b2f_arr_flags.reshape(-1).cpu().numpy()],
        )

    for _k0d_nm in _K0D_SELECTED:
        _rows = []
        _candidate_16_ok = True
        if _k0d_nm == "k0d_f":
            _before = _k0df_state_snapshot()
        elif _k0d_nm == "k0d_sq_b2f":
            _before = _k0dsq_b2f_state_snapshot()
        else:
            _before = None
        for _rep in range(16):
            # Control: no quant, so the finite 0x7E/NaN ingress poison must survive downstream.
            a_src_u8.fill_(0x7E); sc_src.fill_(float("nan")); nvi.zero_(); pull_ptr.zero_(); pull_src.fill_(-1)
            part[:T_LOC_MAX].fill_(1000.0); cand_out.zero_(); perr.zero_(); pperr.zero_()
            torch.cuda.synchronize(); hbarrier()
            k0d_noquant_control_graph.replay(); torch.cuda.synchronize()
            _ctrl_g = gate(cand_out); _ctrl_failed = bool(not ok(_ctrl_g))
            _k0d_control_all_failed = bool(_k0d_control_all_failed and _ctrl_failed)

            # Candidate: re-poison after the control so its captured ingress, plan, and gather-zero
            # are each independently required to write the current epoch's values.
            a_src_u8.fill_(0x7E); sc_src.fill_(float("nan")); nvi.zero_(); pull_ptr.zero_(); pull_src.fill_(-1)
            part[:T_LOC_MAX].fill_(1000.0); cand_out.zero_(); perr.zero_(); pperr.zero_()
            torch.cuda.synchronize(); hbarrier()
            graphs[_k0d_nm].replay(); torch.cuda.synchronize()
            _cand_g = gate(cand_out)
            _cand_perr = int(perr.item()); _cand_pperr = int(pperr.item())
            _cand_nvi = [int(x) for x in nvi.reshape(-1).cpu().numpy()]
            _cand_ok = bool(
                ok(_cand_g) and _cand_perr == 0 and _cand_pperr == 0
                and _cand_nvi == [_padded, T_loc]
            )
            _candidate_16_ok = bool(_candidate_16_ok and _cand_ok)
            _rows.append(dict(
                rep=_rep, control_failed=_ctrl_failed, control_rel_L2=_ctrl_g["rel_L2"],
                candidate_pass=_cand_ok, candidate_rel_L2=_cand_g["rel_L2"],
                candidate_max_abs=_cand_g["max_abs"], plan_err=_cand_perr,
                pperr=_cand_pperr, nvi=_cand_nvi,
            ))
            hbarrier()

        if _k0d_nm == "k0d_f":
            _after = _k0df_state_snapshot()
        elif _k0d_nm == "k0d_sq_b2f":
            _after = _k0dsq_b2f_state_snapshot()
        else:
            _after = None
        _state_ok = True
        _state_delta = None
        if _k0d_nm == "k0d_f":
            _state_delta = dict(
                quant_count=_after["quant_count"] - _before["quant_count"],
                arr_count=_after["arr_count"] - _before["arr_count"],
                don_count=_after["don_count"] - _before["don_count"],
                quant_flags=[a - b for a, b in zip(_after["quant_flags"], _before["quant_flags"])],
                comb_flags=[a - b for a, b in zip(_after["comb_flags"], _before["comb_flags"])],
                arr_flags=[a - b for a, b in zip(_after["arr_flags"], _before["arr_flags"])],
            )
            _state_ok = bool(
                _state_delta["quant_count"] == 16
                and _state_delta["arr_count"] == 16 * 16
                and _state_delta["don_count"] == 16 * 16
                and _state_delta["quant_flags"] == [16] * WORLD
                and _state_delta["comb_flags"] == [16] * WORLD
                and _state_delta["arr_flags"] == [16] * WORLD
            )
        elif _k0d_nm == "k0d_sq_b2f":
            _state_delta = dict(
                b1_count=_after["b1_count"] - _before["b1_count"],
                arr_count=_after["arr_count"] - _before["arr_count"],
                b1_flags=[a - b for a, b in zip(_after["b1_flags"], _before["b1_flags"])],
                arr_flags=[a - b for a, b in zip(_after["arr_flags"], _before["arr_flags"])],
            )
            _state_ok = bool(
                _state_delta["b1_count"] == 16
                and _state_delta["arr_count"] == 16 * 16
                and _state_delta["b1_flags"] == [16] * WORLD
                and _state_delta["arr_flags"] == [16] * WORLD
            )
        _arm_ok = bool(_candidate_16_ok and _state_ok)
        _k0d_epoch_all_local = bool(_k0d_epoch_all_local and _arm_ok)
        _k0d_epoch_rows[_k0d_nm] = dict(
            candidate_16_pass=_candidate_16_ok, state_delta_ok=_state_ok,
            state_before=_before, state_after=_after, state_delta=_state_delta, replays=_rows,
        )

    _k0d_epoch_local = bool(_k0d_epoch_all_local and _k0d_control_all_failed)
    _k0d_epoch_all = _all_ranks_pass(_k0d_epoch_local)
    R["k0d_epoch"] = dict(
        pass_all_ranks=_k0d_epoch_all, local_pass=_k0d_epoch_local,
        noquant_downstream_control_16_fail=_k0d_control_all_failed, arms=_k0d_epoch_rows,
    )
    R["k0d_epoch_ok"] = _k0d_epoch_all
    gate_ok = bool(gate_ok and _k0d_epoch_all)
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(f"[K0D GATE] epoch poison pass={_k0d_epoch_all} arms={_K0D_SELECTED} "
              f"noquant_control_16_fail={_k0d_control_all_failed}", flush=True)
    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== k0d split-quant edge-C wait non-vacuity gate =====================
R["k0d_sq_b2f_wait_control_ok"] = True
if cap_ok and "k0d_sq_b2f" in _K0D_SELECTED:
    pperr.zero_(); cand_out.zero_(); torch.cuda.synchronize(); hbarrier()
    k0d_sq_b2f_wait_control_graph.replay(); torch.cuda.synchronize()
    _k0d_b2f_control_pperr = int(pperr.item())
    _k0d_b2f_control_local = bool(_k0d_b2f_control_pperr & 8)
    _k0d_b2f_control_all = _all_ranks_pass(_k0d_b2f_control_local)
    R["k0d_sq_b2f_wait_control"] = dict(
        pass_all_ranks=_k0d_b2f_control_all,
        local_pass=_k0d_b2f_control_local,
        pperr=_k0d_b2f_control_pperr,
        expected_timeout_bit=8,
    )
    R["k0d_sq_b2f_wait_control_ok"] = _k0d_b2f_control_all
    gate_ok = bool(gate_ok and _k0d_b2f_control_all)
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(
            f"[K0D GATE] split-quant edge-C wait non-vacuity "
            f"pass={_k0d_b2f_control_all} pperr={_k0d_b2f_control_pperr}",
            flush=True,
        )
    pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== distinct-route graph-replay gate =====================
# Every setup and replay phase first reports its local status through gloo.  No rank enters a
# graph replay, all_gather, or symmetric-heap barrier that another rank has collectively rejected.
def _route_collective_phase(name, action):
    local_error = None
    try:
        action()
    except Exception as route_phase_error:
        local_error = "".join(
            traceback.format_exception_only(type(route_phase_error), route_phase_error)
        ).strip()
    errors = [None for _ in range(world)]
    dist.all_gather_object(errors, local_error)
    return all(error is None for error in errors), dict(name=name, errors=errors)


def _pf4h_route_metadata_evidence(route):
    """Validate PF4H's device-produced plan against the swapped host route.

    ``sti`` arrival rows and ``pull_src`` remote row numbers are nondeterministic by design, so
    they have no sound independent byte-for-byte host ordering.  They are instead checked against
    fresh device ``recv_*``/``pull_stage`` values whose expert, weight, fanout, and destination
    metadata are independently derived from the alternate corpus below.  Each relationship is a
    hard runtime gate; an unavailable comparison is therefore never treated as a pass.
    """
    meta = route["pf4h_metadata_after"]
    st = pf3_state["pf4h"]
    expected_blocks = np.asarray(meta["block_totals"], dtype=np.int64)
    expected_block_vector = np.asarray(meta["block_vector"], dtype=np.int64)
    expected_counts = np.asarray(meta["local_expert_rows"], dtype=np.int64)
    expected_tloc = int(meta["arrival_rows"][rank])
    expected_sei = np.repeat(np.arange(E, dtype=np.int32), expected_block_vector[rank])
    expected_padded = int(expected_sei.size * 32)

    actual_nvi = nvi.reshape(-1).cpu().numpy().astype(np.int64)
    actual_padded = int(actual_nvi[0])
    actual_blocks = actual_padded // 32 if actual_padded >= 0 else -1
    recv_eid = st["recv_eid"][:max(0, min(expected_tloc, T_LOC_MAX))].cpu().numpy()
    recv_wgt_bits = st["recv_wgt"][:max(0, min(expected_tloc, T_LOC_MAX))].cpu().numpy().view(np.uint32)
    actual_counts = np.asarray([
        int(np.count_nonzero(recv_eid == rank * E + expert)) for expert in range(E)
    ], dtype=np.int64)
    recv_count_diff = int(np.abs(actual_counts - expected_counts).sum())
    recv_weight_bit_diffs = 0
    for expert in range(E):
        actual_bits = np.sort(recv_wgt_bits[recv_eid == rank * E + expert])
        expected_bits = meta["local_weight_bits"][expert]
        if actual_bits.shape != expected_bits.shape:
            recv_weight_bit_diffs += abs(int(actual_bits.size) - int(expected_bits.size)) + max(
                int(actual_bits.size), int(expected_bits.size)
            )
        else:
            recv_weight_bit_diffs += int(np.count_nonzero(actual_bits != expected_bits))

    inspect_rows = max(0, min(actual_padded, PADMAX))
    actual_sei = sei[:max(0, min(actual_blocks, PADMAX // 32))].reshape(-1).cpu().numpy()
    sei_bit_diffs = (
        abs(int(actual_sei.size) - int(expected_sei.size)) + max(int(actual_sei.size), int(expected_sei.size))
        if actual_sei.shape != expected_sei.shape
        else int(np.count_nonzero(actual_sei != expected_sei))
    )
    sti_u32 = sti[:inspect_rows].reshape(-1).cpu().numpy().view(np.uint32)
    swt_bits = swt[:inspect_rows].reshape(-1).cpu().numpy().view(np.uint32)
    expected_live = np.zeros((inspect_rows,), dtype=bool)
    offset = 0
    for expert, count in enumerate(expected_counts):
        width = int(expected_block_vector[rank, expert] * 32)
        expected_live[offset:min(offset + int(count), inspect_rows)] = True
        offset += width
    arrival = sti_u32 & np.uint32(0x00FFFFFF)
    slot = sti_u32 >> np.uint32(24)
    live_idx = np.flatnonzero(expected_live)
    live_bounds_ok = bool(
        np.all(arrival[live_idx] < expected_tloc) and np.all(slot[live_idx] < TOPK)
    ) if live_idx.size else True
    sti_recv_mismatch = 0
    swt_recv_mismatch = 0
    if live_bounds_ok:
        expected_experts = expected_sei[(live_idx // 32).astype(np.int64)] + rank * E
        recv_ids_at_sti = recv_eid[arrival[live_idx], slot[live_idx]]
        recv_wgt_at_sti = recv_wgt_bits[arrival[live_idx], slot[live_idx]]
        sti_recv_mismatch = int(np.count_nonzero(recv_ids_at_sti != expected_experts))
        swt_recv_mismatch = int(np.count_nonzero(swt_bits[live_idx] != recv_wgt_at_sti))
    else:
        sti_recv_mismatch = swt_recv_mismatch = max(1, int(live_idx.size))
    pad_idx = np.flatnonzero(~expected_live)
    pad_sti = (np.uint32(expected_tloc) & np.uint32(0x00FFFFFF)) | (np.uint32(TOPK) << np.uint32(24))
    pad_sti_mismatch = int(np.count_nonzero(sti_u32[pad_idx] != pad_sti))
    pad_swt_mismatch = int(np.count_nonzero(swt_bits[pad_idx] != 0))

    actual_pull_ptr = pull_ptr[:T + 1].reshape(-1).cpu().numpy().astype(np.int64)
    expected_pull_ptr = meta["local_pull_ptr"]
    pull_ptr_diffs = int(np.count_nonzero(actual_pull_ptr != expected_pull_ptr))
    pull_stage = st["pull_stage"].reshape(T, TOPK, 2).cpu().numpy()
    expected_primary = meta["local_primary"]
    pull_stage_dest_diffs = int(np.count_nonzero(pull_stage[:, :, 0] != expected_primary))
    primary_mask = expected_primary >= 0
    pull_stage_row_bad = 0
    if np.any(primary_mask):
        stage_dest = pull_stage[:, :, 0][primary_mask]
        stage_row = pull_stage[:, :, 1][primary_mask]
        destination_limits = np.asarray(meta["arrival_rows"], dtype=np.int64)[stage_dest]
        pull_stage_row_bad = int(np.count_nonzero((stage_row < 0) | (stage_row >= destination_limits)))
    pull_src_count = int(expected_pull_ptr[-1])
    expected_pull_src = pull_stage[primary_mask].reshape(-1, 2)
    actual_pull_src = pull_src[:pull_src_count].reshape(-1, 2).cpu().numpy()
    pull_src_diffs = (
        abs(int(actual_pull_src.size) - int(expected_pull_src.size))
        + max(int(actual_pull_src.size), int(expected_pull_src.size))
        if actual_pull_src.shape != expected_pull_src.shape
        else int(np.count_nonzero(actual_pull_src != expected_pull_src))
    )
    checks = dict(
        nvi_exact=actual_nvi.tolist() == [expected_padded, expected_tloc],
        nvi=actual_nvi.tolist(), expected_nvi=[expected_padded, expected_tloc],
        aggregate_blocks_exact=actual_blocks == int(expected_blocks[rank]),
        sei_exact=sei_bit_diffs == 0, sei_bit_diffs=sei_bit_diffs,
        recv_counts_exact=recv_count_diff == 0, recv_count_diff=recv_count_diff,
        recv_weights_exact=recv_weight_bit_diffs == 0, recv_weight_bit_diffs=recv_weight_bit_diffs,
        sti_relational_exact=live_bounds_ok and sti_recv_mismatch == 0 and pad_sti_mismatch == 0,
        sti_recv_mismatch=sti_recv_mismatch, pad_sti_mismatch=pad_sti_mismatch,
        swt_relational_exact=swt_recv_mismatch == 0 and pad_swt_mismatch == 0,
        swt_recv_mismatch=swt_recv_mismatch, pad_swt_mismatch=pad_swt_mismatch,
        pull_ptr_exact=pull_ptr_diffs == 0, pull_ptr_diffs=pull_ptr_diffs,
        pull_src_relational_exact=(
            pull_stage_dest_diffs == 0 and pull_stage_row_bad == 0 and pull_src_diffs == 0
        ),
        pull_stage_dest_diffs=pull_stage_dest_diffs, pull_stage_row_bad=pull_stage_row_bad,
        pull_src_diffs=pull_src_diffs,
        sti_swt_ordering="nondeterministic arrival order; checked against fresh recv_* relations",
        pull_src_ordering="nondeterministic remote rows; checked against fresh pull_stage compaction",
    )
    checks["pass_local"] = bool(all(value for key, value in checks.items() if key.endswith("_exact")))
    return checks


R["k0d_route_swap_ok"] = True
_PF4H_ROUTE_SELECTED = "pf4h" in R["arms"]
_ROUTE_SWAP_SELECTED = _K0D_SELECTED + (["pf4h"] if _PF4H_ROUTE_SELECTED else [])
R["pf4h_route_swap_ok"] = not _PF4H_ROUTE_SELECTED
_route_gate_local = dict(
    selected=tuple(_ROUTE_SWAP_SELECTED), cap_ok=bool(cap_ok),
    production_ready=bool("production" in R["arms"] and "production" in graphs and "prod" in _ph),
    corpus_ready=bool(k0d_route and k0d_route.get("collective_ready", False)),
    pf4h_metadata_ready=bool(k0d_route and k0d_route.get("collective_metadata_ready", False)),
    pf4h_safe_direction=bool(k0d_route and k0d_route.get("collective_safe_direction", False)),
)
_route_gate_by_rank = [None for _ in range(world)]
dist.all_gather_object(_route_gate_by_rank, _route_gate_local)
_route_swap_any_selected = any(status["selected"] for status in _route_gate_by_rank)
_route_selection_consistent = all(
    status["selected"] == _route_gate_by_rank[0]["selected"] for status in _route_gate_by_rank
)
_route_arms = list(_route_gate_by_rank[0]["selected"]) if _route_selection_consistent else []
_route_pf4h_selected = "pf4h" in _route_arms
_route_replay_ready = bool(
    _route_selection_consistent
    and all(status["cap_ok"] for status in _route_gate_by_rank)
    and all(status["production_ready"] for status in _route_gate_by_rank)
    and all(status["corpus_ready"] for status in _route_gate_by_rank)
    and (
        not _route_pf4h_selected
        or all(status["pf4h_metadata_ready"] and status["pf4h_safe_direction"]
               for status in _route_gate_by_rank)
    )
)
if _route_swap_any_selected:
    _route_setup_errors = []
    if not _route_selection_consistent:
        _route_setup_errors.append("route-swap arm selection differs across ranks")
    if not all(status["cap_ok"] for status in _route_gate_by_rank):
        _route_setup_errors.append("graph capture failed on at least one rank")
    if not all(status["production_ready"] for status in _route_gate_by_rank):
        _route_setup_errors.append("route-swap requires K0_ARMS to include the production arm on every rank")
    if not all(status["corpus_ready"] for status in _route_gate_by_rank):
        _route_setup_errors.append("alternate route corpus was not ready on every rank")
    if _route_pf4h_selected and not all(
        status["pf4h_metadata_ready"] and status["pf4h_safe_direction"]
        for status in _route_gate_by_rank
    ):
        _route_setup_errors.append("PF4H alternate route lacks collective safe decisive metadata")
    _route_setup_error = "; ".join(_route_setup_errors) or None
    if not _route_replay_ready:
        if _K0D_SELECTED:
            R["k0d_route_swap"] = dict(
                pass_all_ranks=False, ready=False, corpus=K0_ROUTE_SWAP_CORPUS,
                error=_route_setup_error, setup_by_rank=_route_gate_by_rank,
            )
            R["k0d_route_swap_ok"] = False
        if _PF4H_ROUTE_SELECTED:
            R["pf4h_route_swap"] = dict(
                pass_all_ranks=False, ready=False, corpus=K0_ROUTE_SWAP_CORPUS,
                error=_route_setup_error, setup_by_rank=_route_gate_by_rank,
                production_arm_ready=all(status["production_ready"] for status in _route_gate_by_rank),
                blocks_before=None if k0d_route is None else k0d_route.get("pf4h_blocks_before"),
                blocks_after=None if k0d_route is None else k0d_route.get("pf4h_blocks_after"),
            )
            R["pf4h_route_swap_ok"] = False
        gate_ok = False
    else:
        _route_before_local = [int(_padded // 32)]
        _route_local_distinct = [False]
        _route_before_ok, _route_before_phase = _route_collective_phase(
            "capture_original_metadata",
            lambda: (
                _route_before_local.__setitem__(0, int(_padded // 32)),
                _route_local_distinct.__setitem__(0, bool(
                    not bool((topk_ids == k0d_route["topk_ids_alt"]).all().item())
                    or not bool((topk_wgt == k0d_route["topk_wgt_alt"]).all().item())
                )),
            ),
        )
        _route_before = [None for _ in range(world)]
        if _route_before_ok:
            dist.all_gather_object(_route_before, _route_before_local[0])
        _route_phase_log = [_route_before_phase]
        _route_alt_ok = _route_before_ok
        if _route_alt_ok:
            _route_alt_ok, _phase = _route_collective_phase(
                "install_alternate_inputs",
                lambda: (
                    hbarrier(), hidden.copy_(k0d_route["hidden_alt"]),
                    topk_ids.copy_(k0d_route["topk_ids_alt"]), topk_wgt.copy_(k0d_route["topk_wgt_alt"]),
                    perr.zero_(), pperr.zero_(), torch.cuda.synchronize(), hbarrier(),
                ),
            )
            _route_phase_log.append(_phase)
        _route_prod = None
        _route_prod_gate = None
        _route_prod_ok = False
        if _route_alt_ok:
            _route_prod_capture = {}
            def _route_run_production(capture=_route_prod_capture):
                graphs["production"].replay(); torch.cuda.synchronize()
                capture["out"] = _ph["prod"].clone()
                capture["gate"] = gate_against_ref(capture["out"], k0d_route["ref_alt"])
            _route_alt_ok, _phase = _route_collective_phase("alternate_production", _route_run_production)
            _route_phase_log.append(_phase)
            if _route_alt_ok:
                _route_prod = _route_prod_capture["out"]
                _route_prod_gate = _route_prod_capture["gate"]
                _route_prod_ok = bool(ok(_route_prod_gate))
        _route_results = {}
        if _route_alt_ok:
            for _route_nm in _route_arms:
                _route_capture = {}
                def _route_run_candidate(name=_route_nm, capture=_route_capture):
                    cand_out.zero_(); perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
                    graphs[name].replay(); torch.cuda.synchronize()
                    capture["gate"] = gate_against_ref(cand_out, k0d_route["ref_alt"])
                    capture["pair"] = _pair_gate(cand_out, _route_prod)
                    capture["perr"] = int(perr.item()); capture["pperr"] = int(pperr.item())
                    capture["blocks"] = int(nvi.reshape(-1)[0].item()) // 32
                    if name == "pf4h": capture["evidence"] = _pf4h_route_metadata_evidence(k0d_route)
                _candidate_ok, _phase = _route_collective_phase(
                    f"alternate_{_route_nm}", _route_run_candidate
                )
                _route_phase_log.append(_phase)
                if not _candidate_ok:
                    _route_alt_ok = False
                    break
                _route_blocks = [None for _ in range(world)]
                dist.all_gather_object(_route_blocks, _route_capture["blocks"])
                _route_changed = bool(_route_blocks != _route_before)
                _route_rank2_45_to_46 = bool(
                    len(_route_before) > 2 and _route_before[2] == 45 and _route_blocks[2] == 46
                )
                _route_more_blocks = bool(any(
                    after > before for before, after in zip(_route_before, _route_blocks)
                ))
                if _route_nm == "pf4h":
                    _route_meta_ok = bool(
                        _route_before == k0d_route["pf4h_blocks_before"]
                        and _route_blocks == k0d_route["pf4h_blocks_after"]
                        and _route_capture["evidence"]["pass_local"]
                    )
                else:
                    _route_meta_ok = (
                        (_route_changed and _route_more_blocks) if K0_SYNTH_ROUTE
                        else _route_rank2_45_to_46
                        if K0_ROUTE_SWAP_CORPUS.endswith("skewed_20260723T214320Z")
                        else _route_changed
                    )
                _route_arm_ok = bool(
                    ok(_route_capture["gate"])
                    and _route_capture["pair"]["max_abs"] <= 0.02
                    and _route_capture["pair"]["rel_L2"] <= 0.01
                    and _route_capture["pair"]["nonfinite"] == 0
                    and _route_capture["perr"] == 0 and _route_capture["pperr"] == 0
                    and _route_meta_ok
                )
                _route_results[_route_nm] = dict(
                    pass_local=_route_arm_ok, alt_corpus_gate=_route_capture["gate"],
                    vs_production=_route_capture["pair"], plan_err=_route_capture["perr"],
                    pperr=_route_capture["pperr"], blocks_before=_route_before,
                    blocks_after=_route_blocks, metadata_changed=_route_changed,
                    metadata_ok=_route_meta_ok, rank2_45_to_46=_route_rank2_45_to_46,
                    alternate_has_more_blocks=_route_more_blocks,
                    evidence=_route_capture.get("evidence"),
                )

        # Always restore the captured input addresses, even if an alternate replay failed.  A
        # failed graph's monotonic protocol state cannot be safely reset; such a run remains
        # blocked and skips the original-graph restore replays below.
        _route_inputs_restored, _phase = _route_collective_phase(
            "restore_original_inputs",
            lambda: (
                hidden.copy_(k0d_route["hidden_orig"]), topk_ids.copy_(k0d_route["topk_ids_orig"]),
                topk_wgt.copy_(k0d_route["topk_wgt_orig"]), perr.zero_(), pperr.zero_(),
                torch.cuda.synchronize(), hbarrier(),
            ),
        )
        _route_phase_log.append(_phase)
        _route_restore_ok = False
        _route_restore_prod = None
        if _route_alt_ok and _route_inputs_restored:
            _restore_prod_capture = {}
            def _route_restore_production(capture=_restore_prod_capture):
                graphs["production"].replay(); torch.cuda.synchronize()
                capture["gate"] = gate(_ph["prod"])
            _restore_prod_ok, _phase = _route_collective_phase(
                "restore_production", _route_restore_production
            )
            _route_phase_log.append(_phase)
            if _restore_prod_ok:
                _route_restore_prod = _restore_prod_capture["gate"]
                _route_restore_ok = bool(ok(_route_restore_prod))
                for _route_nm in _route_arms:
                    _restore_capture = {}
                    def _route_restore_candidate(name=_route_nm, capture=_restore_capture):
                        cand_out.zero_(); perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
                        graphs[name].replay(); torch.cuda.synchronize()
                        capture["gate"] = gate(cand_out)
                        capture["perr"] = int(perr.item()); capture["pperr"] = int(pperr.item())
                    _restore_arm_ok, _phase = _route_collective_phase(
                        f"restore_{_route_nm}", _route_restore_candidate
                    )
                    _route_phase_log.append(_phase)
                    if not _restore_arm_ok:
                        _route_restore_ok = False
                        break
                    _route_results[_route_nm].update(
                        restore_gate=_restore_capture["gate"],
                        restore_plan_err=_restore_capture["perr"],
                        restore_pperr=_restore_capture["pperr"],
                        restore_pass=bool(
                            ok(_restore_capture["gate"])
                            and _restore_capture["perr"] == 0 and _restore_capture["pperr"] == 0
                        ),
                    )
                    _route_restore_ok = bool(
                        _route_restore_ok and _route_results[_route_nm]["restore_pass"]
                    )

        # Keep the final cleanup in the same error-reporting protocol.  In particular, a restore
        # failure must become a collective timing-gate failure rather than a unilateral exception
        # after peers have proceeded to their next synchronization point.
        _route_finalize_ok, _phase = _route_collective_phase(
            "finalize_route_swap",
            lambda: (perr.zero_(), pperr.zero_(), torch.cuda.synchronize(), hbarrier()),
        )
        _route_phase_log.append(_phase)
        _route_common_local_ok = bool(
            _route_local_distinct[0] and _route_prod_ok and _route_restore_ok
            and _route_inputs_restored and _route_finalize_ok
        )
        if _K0D_SELECTED:
            _k0d_route_arms = {nm: _route_results.get(nm, {}) for nm in _K0D_SELECTED}
            _k0d_route_local_ok = bool(
                _route_common_local_ok and len(_k0d_route_arms) == len(_K0D_SELECTED)
                and all(v.get("pass_local", False) for v in _k0d_route_arms.values())
            )
            _k0d_route_all_ok = _all_ranks_pass(_k0d_route_local_ok)
            R["k0d_route_swap"] = dict(
                pass_all_ranks=_k0d_route_all_ok, local_pass=_k0d_route_local_ok,
                corpus=K0_ROUTE_SWAP_CORPUS, route_inputs_distinct=_route_local_distinct[0],
                production_alt_gate=_route_prod_gate, production_restore_gate=_route_restore_prod,
                inputs_restored=_route_inputs_restored,
                state_restore=("replayed_original_graphs" if _route_restore_ok and _route_finalize_ok
                               else "blocked_after_phase_error"),
                phase_log=_route_phase_log, arms=_k0d_route_arms,
            )
            R["k0d_route_swap_ok"] = _k0d_route_all_ok
            gate_ok = bool(gate_ok and _k0d_route_all_ok)
        if _PF4H_ROUTE_SELECTED:
            _pf4h_route_arm = _route_results.get("pf4h", {})
            _pf4h_route_local_ok = bool(
                _route_common_local_ok and _pf4h_route_arm.get("pass_local", False)
            )
            _pf4h_route_all_ok = _all_ranks_pass(_pf4h_route_local_ok)
            R["pf4h_route_swap"] = dict(
                pass_all_ranks=_pf4h_route_all_ok, local_pass=_pf4h_route_local_ok,
                ready=True, safe_direction=True, corpus=K0_ROUTE_SWAP_CORPUS,
                route_inputs_distinct=_route_local_distinct[0], production_alt_gate=_route_prod_gate,
                production_restore_gate=_route_restore_prod, inputs_restored=_route_inputs_restored,
                state_restore=("replayed_original_graphs" if _route_restore_ok and _route_finalize_ok
                               else "blocked_after_phase_error"),
                phase_log=_route_phase_log, metadata_ok=_pf4h_route_arm.get("metadata_ok", False),
                evidence=_pf4h_route_arm.get("evidence"), arm=_pf4h_route_arm,
            )
            R["pf4h_route_swap_ok"] = _pf4h_route_all_ok
            gate_ok = bool(gate_ok and _pf4h_route_all_ok)
        if rank == 0:
            if _K0D_SELECTED:
                print(f"[K0D GATE] distinct-route replay pass={R['k0d_route_swap_ok']}", flush=True)
            if _PF4H_ROUTE_SELECTED:
                print(f"[PF4H GATE] distinct-route replay pass={R['pf4h_route_swap_ok']} "
                      f"safe_direction={k0d_route['collective_safe_direction']}", flush=True)

# ===================== n2r two-sided EPOCH gate (§L72 instrument — runs AFTER capture) =====================
# (a) EPOCH-POSITIVE: 16 poisoned replays of each selected n2r graph. k0pf2 arms already run this
#     positive gate in the k0pf2 block above; frozen_n2r runs it here. Before each replay poison the
#     cross-rank SOURCE buffers (a_src/sc_src <- 0x7E / NaN) and the live symmetric partial sink.
#     A racy arm reads the finite sentinel/zero -> rel_L2 explodes; a correct arm reads fresh data.
# (b) EPOCH-CONTROL (fail-closed, DEDICATED never-signaled buffers): grid_ctrl = 256 (epoch 1) with
#     flags_ctrl = 0 -> the combine_wait spin can NEVER be satisfied -> bounded spin times out ->
#     pperr != 0 MUST be observed. The timeout path records the error and then continues, so the
#     already-valid partial payload may still produce a numerically valid output; pperr is the
#     deterministic non-vacuity signal.
R["n2r_epoch_ok"] = True
_selected_n2r = [nm for nm in ("frozen_n2r", "pf2_n2r") if nm in R["arms"]]
if cap_ok and _selected_n2r and _N2R_OK:
    if "frozen_n2r" in R["arms"]:
        _ep_ok = True
        for rep in range(16):
            a_src_u8.fill_(0x7E); sc_src.fill_(float("nan"))
            # Poison the LIVE symmetric partial rows, not the empty capacity tail.
            # The graph clears/rebuilds its own local rows. A premature remote
            # combine therefore observes either this finite sentinel or the
            # graph's intermediate zeroes instead of a prior replay's valid data.
            part[:T_LOC_MAX].fill_(1.0e3)
            torch.cuda.synchronize(); hbarrier()
            cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize()
            graphs["frozen_n2r"].replay(); torch.cuda.synchronize()
            g = gate(cand_out); pe = int(pperr.cpu().numpy().reshape(-1)[0])
            if not ok(g) or pe != 0:
                _ep_ok = False
                R[f"n2r_epoch_rep{rep}"] = dict(rel_L2=g["rel_L2"], pperr=pe)
    else:
        # The k0pf2 block above already ran 16 poisoned replays of pf2_n2r.
        _ep_ok = bool(R.get("k0pf2_epoch_ok", False))
    R["n2r_epoch_positive"] = _ep_ok
    n2r_flags_ctrl, n2r_flags_ctrl_p = mori_t((WORLD, 1), "int32"); n2r_flags_ctrl.zero_()
    n2r_grid_ctrl = torch.full((1,), 256, dtype=torch.int32, device="cuda")   # epoch 1, flags 0 -> unsatisfiable
    def _nowait_body(stream):
        _plan(stream); _quant(stream); _gather_pull(stream)
        part[:T_LOC_MAX].zero_()
        k0_n2.n2_phase1(a_dst, sc_dst, w13, fc1_scale, sti1, sei1, nvi1, _n2_bufs["a2q"], _n2_bufs["dq2"], stream)
        k0_n2.n2_phase2(_n2_bufs["a2q"], _n2_bufs["dq2"], w2c, fc2_scale, sti1, swt1, sei1, nvi1, part, stream)
        fn_combine_wait.launch((T,), (CBLK,), 0, stream,
                               cand_out.data_ptr(), part_p, pull_ptr.data_ptr(), pull_src.data_ptr(),
                               stage.data_ptr(), n2r_flags_ctrl_p, n2r_grid_ctrl.data_ptr(), pperr.data_ptr(),
                               int(os.environ.get("K0_SPIN_LIMIT_CTRL", "200000")), T, H, rank, WORLD)
    hbarrier(); cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize()
    _nowait_body(sp()); torch.cuda.synchronize(); hbarrier()
    g_ctrl = gate(cand_out); pe_ctrl = int(pperr.cpu().numpy().reshape(-1)[0])
    R["n2r_epoch_control"] = dict(rel_L2=g_ctrl["rel_L2"], pperr=pe_ctrl)
    R["n2r_epoch_control_fails"] = bool(pe_ctrl != 0)
    _n2r_epoch_local = bool(_ep_ok and R["n2r_epoch_control_fails"])
    R["n2r_epoch_ok"] = _all_ranks_pass(_n2r_epoch_local)
    gate_ok = bool(gate_ok and R["n2r_epoch_ok"])
    R["timing_gate_ok"] = gate_ok
    if rank == 0:
        print(f"[MARK] n2r_epoch_positive={_ep_ok} (16 poisoned replays) | "
              f"n2r_epoch_control pperr={pe_ctrl} rel_L2={g_ctrl['rel_L2']:.5f} "
              f"fails_as_required={R['n2r_epoch_control_fails']} | n2r_epoch_ok={R['n2r_epoch_ok']}", flush=True)
    hbarrier()
    pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== repeated graph-safety stress (HARD gate) =====================
# Unsynced back-to-back replays then re-check. Any numerical failure or fail-closed
# pperr signal blocks timing; a stress failure cannot be downgraded to telemetry.
if cap_ok:
    STRESS = int(os.environ.get("K0_STRESS", "30"))
    all_stress = True
    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
    for name, _, buf in ARMS:
        hbarrier()
        perr.zero_(); pperr.zero_(); torch.cuda.synchronize()
        for _ in range(STRESS): graphs[name].replay()
        torch.cuda.synchronize(); hbarrier()
        if name not in _PROD_LIKE: cand_out.zero_()
        torch.cuda.synchronize(); hbarrier(); graphs[name].replay(); torch.cuda.synchronize()
        _stress_pperr = int(pperr.cpu().numpy().reshape(-1)[0])
        _stress_perr = int(perr.cpu().numpy().reshape(-1)[0]) if name in _CUSTOM_PLAN_ARMS else 0
        R["graph"][name]["stress_pass"] = bool(
            ok_arm(name, gate(_obuf(name))) and _stress_pperr == 0 and _stress_perr == 0
        )
        R["graph"][name]["stress_pperr"] = _stress_pperr
        R["graph"][name]["stress_plan_err"] = _stress_perr
        hbarrier()
        if not R["graph"][name]["stress_pass"]: all_stress = False
    R["pperr_after_stress"] = int(pperr.cpu().numpy().reshape(-1)[0])
    _stress_local = bool(all_stress and R["pperr_after_stress"] == 0)
    R["stress_all_pass"] = _all_ranks_pass(_stress_local)
    if rank == 0: print(f"[MARK] stress done stress_all_pass={R['stress_all_pass']} pperr={R['pperr_after_stress']} (HARD gate)", flush=True)

# ===================== interleaved cross-graph coupling stress (Codex Q1 — HARD gate) =====================
# The grouped stress above (30x A then 30x F) does NOT test the shared-MORI-op coupling. This does: alternate
# A->F->A and F->A->F, validating EACH arm IMMEDIATELY after its replay (op.combine's output view is overwritten
# by the next replay). A wedge here = shared-op graph coupling is unsafe -> BLOCK (the external timeout catches
# a collective deadlock, which will not throw). Same replay order on every rank (globally deterministic).
xstress_pass = None
if cap_ok and len(R["arms"]) > 1:
    _xnames = list(R["arms"])
    _xr = True
    perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
    for _c in range(int(os.environ.get("K0_XSTRESS", "25"))):
        _rot = _c % len(_xnames)
        _order = _xnames[_rot:] + _xnames[:_rot]
        if _c & 1:
            _order = list(reversed(_order))
        for nm in _order:
            if nm not in _PROD_LIKE: cand_out.zero_()
            torch.cuda.synchronize(); dist.barrier()
            graphs[nm].replay(); torch.cuda.synchronize()
            _xok = ok_arm(nm, gate(_obuf(nm)))
            if nm in _CUSTOM_PLAN_ARMS:
                _xok = bool(_xok and int(perr.cpu().numpy().reshape(-1)[0]) == 0)
            _xok = bool(_xok and int(pperr.cpu().numpy().reshape(-1)[0]) == 0)
            if not _xok:
                _xr = False
            dist.barrier()
    xstress_pass = _all_ranks_pass(_xr); R["xstress_pass"] = xstress_pass
    if rank == 0:
        print(f"[K0PF GATE] interleaved {len(_xnames)}-graph stress pass={xstress_pass} "
              f"(HARD gate)", flush=True)

# FINAL timing gate: capture + per-arm graph gates + generic control-fails + ALL bridge validations +
# interleaved cross-graph coupling. Any False -> no timing (report BLOCKED with the failing gate).
gate_ok = bool(
    gate_ok
    and bridge_val_ok
    and R.get("stress_all_pass", False)
    and R.get("pperr_after_stress", 1) == 0
    and (xstress_pass in (True, None))
    and R.get("c32_gate_ok", True)
    and R.get("ah_gate_ok", True)
    and R.get("n2_gate_ok", True)
    and R.get("k0d_epoch_ok", True)
    and R.get("k0d_sq_b2f_wait_control_ok", True)
    and R.get("k0d_route_swap_ok", True)
    and R.get("pf4h_route_swap_ok", True)
)
R["timing_gate_ok"] = gate_ok
if rank == 0: print(f"[MARK] FINAL timing_gate_ok={gate_ok} (cap={cap_ok} bridge_val={bridge_val_ok} xstress={xstress_pass} c32={R.get('c32_gate_ok')} ah={R.get('ah_gate_ok')} n2={R.get('n2_gate_ok')})", flush=True)

# ===================== k0d decode diagnostics (ADDITIVE; run after every hard gate passes) =====================
# Neither diagnostic is an A/B arm. k0d_db_micro owns its flag slices and never touches arm
# protocol state; k0d_mega_ts advances the shared pd/mega monotonic state only by complete
# epochs, exactly like one extra replay of the k0d_mega arm. Both fail closed on pperr.
if (
    gate_ok
    and (_K0_DB_MICRO or _K0_MEGA_TS or _K0_PF3_TS)
    and fn_k0d_phase_timer_calibration is not None
    and (
        (_K0_DB_MICRO and fn_k0d_db_micro is None)
        or (_K0_MEGA_TS and fn_k0d_mega_ts is None)
        or (_K0_PF3_TS and (fn_pf3_qpush_ts is None or fn_pf3_dsort_ts is None))
        or (_K0_PF4_TS and fn_pf4_dsort_ts is None)
    )
):
    raise RuntimeError(
        "diagnostic kernels requested but not loaded: "
        f"db_micro={_K0_DB_MICRO}/{fn_k0d_db_micro is not None} "
        f"mega_ts={_K0_MEGA_TS}/{fn_k0d_mega_ts is not None} "
        f"pf3_ts={_K0_PF3_TS}/{fn_pf3_qpush_ts is not None and fn_pf3_dsort_ts is not None} "
        f"pf4_ts={_K0_PF4_TS}/{fn_pf4_dsort_ts is not None}"
    )
if (
    gate_ok
    and (_K0_DB_MICRO or _K0_MEGA_TS or _K0_PF3_TS)
    and fn_k0d_phase_timer_calibration is not None
):
    _diag_tick_us = None
    _cal_stamps = torch.zeros((2,), dtype=torch.int64, device="cuda")
    _tick_us_samples = []
    for _ in range(5):
        _cb, _ce = torch.cuda.Event(True), torch.cuda.Event(True)
        _cb.record()
        fn_k0d_phase_timer_calibration.launch(
            (1,), (1,), 0, sp(), _cal_stamps.data_ptr(), 1000000
        )
        _ce.record()
        torch.cuda.synchronize()
        _cal_ticks = int(_cal_stamps[1].item() - _cal_stamps[0].item())
        _cal_us = float(_cb.elapsed_time(_ce) * 1e3)
        if _cal_ticks > 0:
            _tick_us_samples.append(_cal_us / _cal_ticks)
    if _tick_us_samples:
        _diag_tick_us = float(np.median(_tick_us_samples))
    hbarrier()

    if _K0_DB_MICRO and fn_k0d_db_micro is not None:
        _micro_reps = int(os.environ.get("K0_DB_MICRO_REPS", "64"))
        if 128 + 12 * _micro_reps > 1024:
            raise ValueError(f"K0_DB_MICRO_REPS={_micro_reps} overflows the 1024-slot result buffer")
        _micro_flags, _micro_flags_p = mori_t((40, 1), "int32")
        _micro_flags.zero_()
        _micro_lscr = torch.zeros((4,), dtype=torch.int32, device="cuda")
        _micro_res = torch.zeros((1024,), dtype=torch.int64, device="cuda")
        pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        fn_k0d_db_micro.launch(
            (1,), (64,), 0, sp(), _micro_flags_p, _micro_lscr.data_ptr(),
            _micro_res.data_ptr(), pperr.data_ptr(), SPIN, rank, WORLD, _micro_reps,
        )
        torch.cuda.synchronize(); hbarrier()
        _micro_pperr = int(pperr.item())
        R["k0d_db_micro"] = dict(
            reps=_micro_reps, tick_us=_diag_tick_us,
            variants=("serial_atomic8", "lanes_atomic8", "serial_store8", "lanes_store8"),
            res=[int(x) for x in _micro_res.cpu().tolist()],
            pperr=_micro_pperr,
            pass_all_ranks=_all_ranks_pass(_micro_pperr == 0),
        )
        if rank == 0:
            print(f"[K0D DB MICRO] reps={_micro_reps} pperr={_micro_pperr} "
                  f"pass={R['k0d_db_micro']['pass_all_ranks']}", flush=True)
        pperr.zero_(); torch.cuda.synchronize(); hbarrier()

    if _K0_MEGA_TS and fn_k0d_mega_ts is not None:
        _ts_reps = int(os.environ.get("K0_MEGA_TS_REPS", "50"))
        # 96 slots = schema v2 (K0DM_TS_SLOTS in k0d_mega_ts.hip): the v1 0..58 map is unchanged,
        # 59 carries the schema literal, and 64..87 carry the per-source M3 arrival stamps.
        _mega_ts_buf = torch.zeros((96,), dtype=torch.int64, device="cuda")
        # (1) eager self-consistency against the unmodified k0d_mega arm body (same algorithm
        # plus stamps; frozen tolerance, never widened).
        cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        k0d_mega_body(sp()); torch.cuda.synchronize(); hbarrier()
        _ts_ref = cand_out.clone()
        cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        k0d_mega_ts_body(sp()); torch.cuda.synchronize(); hbarrier()
        _ts_pair = _pair_gate(cand_out, _ts_ref, live=T)
        _ts_eager_pperr = int(pperr.item())
        _ts_self_ok = bool(
            _ts_pair["max_abs"] <= 0.02 and _ts_pair["rel_L2"] <= 0.01
            and _ts_pair["nonfinite"] == 0 and _ts_eager_pperr == 0
        )
        # (2) diagnostic graph capture (same pattern as the device phase-marker profile).
        _ts_graph = torch.cuda.CUDAGraph()
        _ts_stream = torch.cuda.Stream()
        with torch.cuda.stream(_ts_stream):
            for _ in range(3):
                k0d_mega_ts_body(_ts_stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(_ts_stream)
        torch.cuda.synchronize(); hbarrier()
        with torch.cuda.graph(_ts_graph, stream=_ts_stream):
            k0d_mega_ts_body(torch.cuda.current_stream().cuda_stream)
        hbarrier()
        # (3) replays with per-replay host reads of the stamp buffer.
        _ts_ticks = []
        _ts_pperrs = []
        _ts_broken = False
        for _ in range(_ts_reps):
            pperr.zero_(); _mega_ts_buf.zero_(); torch.cuda.synchronize(); hbarrier()
            _ts_graph.replay(); torch.cuda.synchronize()
            _row = [int(x) for x in _mega_ts_buf.cpu().tolist()]
            if (
                _row[0] == 0
                or any(_row[17 + b] < _row[1 + b] for b in range(16))
                or any(
                    _row[i] < _row[i - 1]
                    for i in (1, 33, 52, 56, 57, 58)
                    if _row[i] != 0
                )
            ):
                _ts_broken = True
            _ts_ticks.append(_row)
            _ts_pperrs.append(int(pperr.item()))
            hbarrier()
        _ts_ok = bool(_ts_self_ok and not _ts_broken and all(x == 0 for x in _ts_pperrs))
        R["k0d_mega_ts"] = dict(
            reps=_ts_reps, tick_us=_diag_tick_us,
            stamp_map=(
                "0:entry 1..16:perCTA_M1done 17..32:perCTA_M2arrive 33:pre_fanout "
                "34..41:per_poke 44..51:per_source_waitpass 52:acquire_done "
                "53:cta0_plan 54:cta1_csr 55:cta15_zero 56:m5_done 57:m6_done 58:end "
                "59:schema(data) 64..71:per_source_pollenter 72..79:per_source_arrive "
                "80..87:per_source_pollspins(data)"
            ),
            ts_schema=2,
            self_check=_ts_pair, eager_pperr=_ts_eager_pperr, replay_pperr=_ts_pperrs,
            ticks=_ts_ticks,
            pass_all_ranks=_all_ranks_pass(_ts_ok),
        )
        if rank == 0:
            print(f"[K0D MEGA TS] reps={_ts_reps} self_ok={_ts_self_ok} "
                  f"pass={R['k0d_mega_ts']['pass_all_ranks']}", flush=True)
        pperr.zero_(); torch.cuda.synchronize(); hbarrier()

    if _K0_PF3_TS:
        _ts_reps = _K0_PF3_TS_REPS
        _pf3_qts_buf = torch.zeros((9248,), dtype=torch.int64, device="cuda")
        # 256 slots = schema v2 (K0P3T_TS_SLOTS in k0pf3_dsort_ts.hip): the v1 0..208 map is
        # unchanged, 11 carries the schema literal, 224..247 the per-source P0 arrival stamps.
        # 384 slots = schema 4 (K0P4T_TS_SLOTS in k0pf4_dsort_ts.hip) under K0_PF4_TS.
        _pf3_dts_buf = torch.zeros((_PF_TS_SLOTS,), dtype=torch.int64, device="cuda")
        _cand_out_ts = torch.zeros_like(cand_out)
        _stage_ts = torch.zeros_like(stage)
        # (1) eager self-consistency: the timestamped twins must reproduce the pf3_pd arm's
        # output within the frozen campaign tolerance (same algorithm plus stamps).
        cand_out.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        pf3_pd_body(sp()); torch.cuda.synchronize(); hbarrier()
        _ts_ref = cand_out.clone()
        pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        pf3_pd_ts_body(sp(), out=_cand_out_ts, stage_buf=_stage_ts)
        torch.cuda.synchronize(); hbarrier()
        _ts_pair = _pair_gate(_cand_out_ts, _ts_ref, live=T)
        _ts_eager_pperr = int(pperr.item())
        _ts_self_ok = bool(
            _ts_pair["max_abs"] <= 0.02 and _ts_pair["rel_L2"] <= 0.01
            and _ts_pair["nonfinite"] == 0 and _ts_eager_pperr == 0
        )
        # (2) diagnostic graph capture (identical pattern to the decode ts diagnostic).
        _ts_graph = torch.cuda.CUDAGraph()
        _ts_stream = torch.cuda.Stream()
        with torch.cuda.stream(_ts_stream):
            for _ in range(3):
                pf3_pd_ts_body(_ts_stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(_ts_stream)
        torch.cuda.synchronize(); hbarrier()
        with torch.cuda.graph(_ts_graph, stream=_ts_stream):
            pf3_pd_ts_body(torch.cuda.current_stream().cuda_stream)
        hbarrier()
        # (3) replays with per-replay host reads of both stamp buffers.
        _ts_rows = []
        _ts_pperrs = []
        for _ in range(_ts_reps):
            pperr.zero_(); _pf3_qts_buf.zero_(); _pf3_dts_buf.zero_()
            torch.cuda.synchronize(); hbarrier()
            _ts_graph.replay(); torch.cuda.synchronize()
            _ts_rows.append(dict(
                q=[int(x) for x in _pf3_qts_buf.cpu().tolist()],
                d=[int(x) for x in _pf3_dts_buf.cpu().tolist()],
            ))
            _ts_pperrs.append(int(pperr.item()))
            hbarrier()
        _ts_ok = bool(_ts_self_ok and all(x == 0 for x in _ts_pperrs))
        R["k0_pf3_ts"] = dict(
            reps=_ts_reps, tick_us=_diag_tick_us,
            grid_qpush=PF3_QPUSH_GRID, grid_dsort=PF3_DSORT_GRID,
            stamp_map_qpush=(
                "0:entry 1..4096:perWarp_quant_done 4097..8192:perWarp_push_done "
                "8193..9216:perCTA_arrive_done 9217:pre_fanout 9218..9225:per_poke"
            ),
            stamp_map_dsort=(
                (
                    "0:entry 9:acquire_done 10:T_loc(data) 11:schema(data) "
                    "16..79:perCTA_P1a_done 80..143:perCTA_P1b_exit "
                    "144..207:perCTA_P1c_done 208..271:perCTA_P2_exit "
                    "272..335:perCTA_P3_done 336:end "
                    "352..359:per_source_pollenter 360..367:per_source_arrive "
                    "368..375:per_source_pollspins(data)"
                ) if _K0_PF4_TS else (
                    "0:entry 1..8:per_source_waitpass 9:acquire_done 10:T_loc(data) "
                    "11:schema(data) 16..79:perCTA_P1done 80..143:perCTA_P2exit "
                    "144..207:perCTA_P3done 208:end 224..231:per_source_pollenter "
                    "232..239:per_source_arrive 240..247:per_source_pollspins(data)"
                )
            ),
            dsort_twin=("k0pf4_dsort_ts" if _K0_PF4_TS else "k0pf3_dsort_ts"),
            grid_dsort_used=(PF4_DSORT_GRID if _K0_PF4_TS else PF3_DSORT_GRID),
            # The §5.2 / #22 split, per replay, in ticks. doorbell_spin is the wait charged to
            # dsort's head; sort_work is everything after the last source arrived.
            split=([
                dict(
                    doorbell_spin=(max(r["d"][360:368]) - r["d"][0]),
                    sort_work=(r["d"][336] - max(r["d"][360:368])),
                    total=(r["d"][336] - r["d"][0]),
                    t_loc=r["d"][10],
                    poll_spins=r["d"][368:376],
                )
                for r in _ts_rows
                if r["d"][11] == 4 and r["d"][336] > 0 and max(r["d"][360:368]) > 0
            ] if _K0_PF4_TS else []),
            ts_schema=_PF_TS_SCHEMA,
            self_check=_ts_pair, eager_pperr=_ts_eager_pperr, replay_pperr=_ts_pperrs,
            rows=_ts_rows,
            pass_all_ranks=_all_ranks_pass(_ts_ok),
        )
        if rank == 0:
            print(f"[K0 PF{4 if _K0_PF4_TS else 3} TS] reps={_ts_reps} "
                  f"twin={R['k0_pf3_ts']['dsort_twin']} schema={_PF_TS_SCHEMA} "
                  f"self_ok={_ts_self_ok} pass={R['k0_pf3_ts']['pass_all_ranks']}", flush=True)
        # The §5.2 / #22 answer. Printed per rank because the doorbell wait is a per-rank
        # quantity (a cool rank waits for the hot one), and the aligned MAX is what the
        # region pays. tick_us converts s_memrealtime ticks with the calibrated constant.
        if _K0_PF4_TS and R["k0_pf3_ts"]["split"]:
            _sp = R["k0_pf3_ts"]["split"]
            _tk = _diag_tick_us if _diag_tick_us else 0.0
            def _med(key):
                v = sorted(x[key] for x in _sp)
                return v[len(v) // 2]
            _db, _sw, _tt = _med("doorbell_spin"), _med("sort_work"), _med("total")
            _frac = (100.0 * _db / _tt) if _tt else 0.0
            R["k0_pf3_ts"]["split_median"] = dict(
                doorbell_spin_ticks=_db, sort_work_ticks=_sw, total_ticks=_tt,
                doorbell_spin_us=(_db * _tk), sort_work_us=(_sw * _tk),
                total_us=(_tt * _tk), doorbell_pct=_frac, n=len(_sp),
            )
            _msg = (f"[K0 PF4 TS SPLIT rank {rank}] n={len(_sp)} "
                    f"doorbell_spin={_db * _tk:.1f}us sort_work={_sw * _tk:.1f}us "
                    f"total={_tt * _tk:.1f}us doorbell={_frac:.1f}%")
            print(_msg, flush=True)
        pperr.zero_(); torch.cuda.synchronize(); hbarrier()

# ===================== N-way position-balanced timing (aligned MAX(rank)); WARM + COLD conditions =====================
def gmax_arr(a):
    t = torch.tensor(a, device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.cpu().numpy()
def p(a, q): return float(np.percentile(a, q))
if gate_ok and not _K0_SKIP_TIMED:
    names = [a[0] for a in ARMS]; n = len(names)
    G = [graphs[nm] for nm in names]
    # ---- cache-conditioning read-sweep scrub graph (Codex E): a deterministic FULL READ of a disjoint
    #      buffer >> MALL/L2, with a tiny device sink. NOT a memset (a memset may take a cache-bypassing
    #      write path and fail to evict). Replayed before EACH timed A/F so back-to-back same-layer L2/MALL
    #      residue cannot flatter one arm. Scrub is OUTSIDE the event-record window -> never in the number.
    SCRUB_MB = int(os.environ.get("K0_SCRUB_MB", "1024"))
    scrub_buf = torch.empty(SCRUB_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev).uniform_(-1.0, 1.0)
    scrub_sink = torch.zeros(1, dtype=torch.float32, device=dev)
    gscrub = torch.cuda.CUDAGraph(); _ss = torch.cuda.Stream()
    with torch.cuda.stream(_ss):
        for _ in range(3): scrub_sink.add_(scrub_buf.sum())
    torch.cuda.current_stream().wait_stream(_ss); torch.cuda.synchronize(); hbarrier()
    with torch.cuda.graph(gscrub, stream=_ss):
        scrub_sink.add_(scrub_buf.sum())
    R["scrub_mb"] = SCRUB_MB

    def run_condition(cold, tag):
        # Fail closed on the exact timed replay outputs. The output check starts only
        # after the stop event has completed, so it is excluded from the measured
        # region while still validating every replay that contributes a sample.
        _timed_gate = {
            nm: dict(
                replays=0, failures=0, worst_rel_L2=0.0, worst_max_abs=0.0,
                max_nonfinite=0, max_pperr=0, max_plan_err=0,
            )
            for nm in names
        }
        perr.zero_(); pperr.zero_(); torch.cuda.synchronize(); hbarrier()
        for _ in range(NUNTIMED):                            # 20 untimed rotations (warm the pipeline)
            for k in range(n):
                if cold: gscrub.replay()
                G[k].replay()
        torch.cuda.synchronize(); hbarrier()
        times = {nm: np.zeros(NTIMED) for nm in names}
        evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
        for i in range(NTIMED):
            order = [(i + j) % n for j in range(n)]          # rotate launch position -> balance each arm over all n slots
            for k in order:
                dist.barrier()
                if cold:
                    gscrub.replay(); torch.cuda.synchronize()   # condition L2/MALL cold
                    dist.barrier()                              # realign ranks AFTER the scrub, BEFORE the start event
                s, e = evs[k]; s.record(); G[k].replay(); e.record()
                torch.cuda.synchronize()
                times[names[k]][i] = s.elapsed_time(e) * 1e3    # us
                _tnm = names[k]
                _tg = gate(_obuf(_tnm))
                _tpperr = int(pperr.cpu().numpy().reshape(-1)[0])
                _tperr = (
                    int(perr.cpu().numpy().reshape(-1)[0])
                    if _tnm in _CUSTOM_PLAN_ARMS else 0
                )
                _tok = bool(ok_arm(_tnm, _tg) and _tpperr == 0 and _tperr == 0)
                _tr = _timed_gate[_tnm]
                _tr["replays"] += 1
                _tr["failures"] += int(not _tok)
                _tr["worst_rel_L2"] = max(_tr["worst_rel_L2"], _tg["rel_L2"])
                _tr["worst_max_abs"] = max(_tr["worst_max_abs"], _tg["max_abs"])
                _tr["max_nonfinite"] = max(_tr["max_nonfinite"], _tg["nonfinite"])
                _tr["max_pperr"] = max(_tr["max_pperr"], _tpperr)
                _tr["max_plan_err"] = max(_tr["max_plan_err"], _tperr)
        _timed_local_ok = all(v["failures"] == 0 for v in _timed_gate.values())
        _timed_all_ok = _all_ranks_pass(_timed_local_ok)
        R[f"timed_replay_gate_{tag}"] = dict(
            pass_all_ranks=_timed_all_ok,
            local_pass=_timed_local_ok,
            arms=_timed_gate,
        )
        if rank == 0:
            _timed_failures = {
                nm: _timed_gate[nm]["failures"] for nm in names
            }
            print(
                f"[K0PF GATE] timed_replays {tag} pass={_timed_all_ok} "
                f"failures={_timed_failures}",
                flush=True,
            )
        if not _timed_all_ok:
            R["timing_gate_ok"] = False
            raise RuntimeError(f"{tag} timed-replay correctness gate failed")
        # Codex caution: aligned-MAX is nonlinear (aliasing can change the critical rank), so persist THIS rank's
        # OWN raw per-iteration times for the fp arms -> per-rank paired (real-alias) deltas computed post-hoc.
        R.setdefault(f"fp_raw_{tag}", {})
        for nm in names:
            if nm in _FP_ARMS: R[f"fp_raw_{tag}"][nm] = times[nm].tolist()
        return {nm: gmax_arr(times[nm]) for nm in names}        # aligned MAX(rank) per iteration

    def summarize(aligned, tag):
        R[f"timing_us_{tag}"] = {nm: dict(p50=p(aligned[nm], 50), p95=p(aligned[nm], 95)) for nm in names}
        prod = aligned["production"]
        def ratio(a, b): r = a / b; return dict(p50=p(r, 50), p95=p(r, 95))
        RT = {f"{nm}/production": ratio(aligned[nm], prod)
              for nm in names if nm != "production"}
        if "k0d_sq_b2f" in names and "k0d_sq" in names:
            RT["k0d_sq_b2f/k0d_sq"] = ratio(aligned["k0d_sq_b2f"], aligned["k0d_sq"])
        if "pf4h" in names and "pf3_pd" in names:
            # pf4h differs from pf3_pd ONLY in the destination sort's count phase, so the
            # per-iteration paired ratio is the isolated hierarchical-histogram delta.
            RT["pf4h/pf3_pd"] = ratio(aligned["pf4h"], aligned["pf3_pd"])
        if "frozen_n2r" in names: RT["frozen_n2r/production"] = ratio(aligned["frozen_n2r"], prod)   # THE N2R REGION HEADLINE
        if "frozen_n2r" in names and "frozen_n2" in names:
            RT["frozen_n2r/frozen_n2"] = ratio(aligned["frozen_n2r"], aligned["frozen_n2"])
        if "frozen_n2r" in names and "frozen_pull" in names:
            RT["frozen_n2r/frozen_pull"] = ratio(aligned["frozen_n2r"], aligned["frozen_pull"])   # n2 GEMM + barrier fold vs our old region
        if "compute_swap_n2" in names: RT["compute_swap_n2/production"] = ratio(aligned["compute_swap_n2"], prod)   # THE N2 HEADLINE
        if "compute_swap_n2" in names and "compute_swap" in names:
            RT["compute_swap_n2/compute_swap"] = ratio(aligned["compute_swap_n2"], aligned["compute_swap"])   # two-phase redesign ALONE (same-session)
        if "compute_swap" in names: RT["compute_swap/production"] = ratio(aligned["compute_swap"], prod)   # oracle n1g headline / reproduction check
        if "compute_swap_c32" in names: RT["compute_swap_c32/production"] = ratio(aligned["compute_swap_c32"], prod)   # THE c32 HEADLINE
        if "compute_swap_c32" in names and "compute_swap" in names:
            RT["compute_swap_c32/compute_swap"] = ratio(aligned["compute_swap_c32"], aligned["compute_swap"])   # tail-aware skip ALONE (same-session)
        if "compute_swap_ah" in names: RT["compute_swap_ah/production"] = ratio(aligned["compute_swap_ah"], prod)   # THE AH HEADLINE (does the W13 addr hoist close the gap?)
        if "compute_swap_ah" in names and "compute_swap" in names:
            RT["compute_swap_ah/compute_swap"] = ratio(aligned["compute_swap_ah"], aligned["compute_swap"])   # W13 address hoist ALONE (same-session; Codex >=3us kill)
        if "compute_swap_ecs" in names: RT["compute_swap_ecs/production"] = ratio(aligned["compute_swap_ecs"], prod)   # THE ECS HEADLINE (does removing the DRAM re-read close the gap?)
        if "compute_swap_ecs" in names and "compute_swap" in names:
            RT["compute_swap_ecs/compute_swap"] = ratio(aligned["compute_swap_ecs"], aligned["compute_swap"])   # expert-major co-scheduling ALONE (same-session; Codex >=3us kill)
        if "compute_swap_ecs_d2" in names: RT["compute_swap_ecs_d2/production"] = ratio(aligned["compute_swap_ecs_d2"], prod)   # THE DECISIVE ECS-D2 HEADLINE
        if "compute_swap_ecs_d2" in names and "compute_swap" in names:
            RT["compute_swap_ecs_d2/compute_swap"] = ratio(aligned["compute_swap_ecs_d2"], aligned["compute_swap"])   # D=2 co-scheduling ALONE (same-session; Codex >=3us kill)
        if "frozen_pull" in names:  RT["frozen_pull/production"]  = ratio(aligned["frozen_pull"], prod)    # Stage-1 context
        if "compute_swap" in names and "frozen_pull" in names:
            RT["compute_swap/frozen_pull"] = ratio(aligned["compute_swap"], aligned["frozen_pull"])
        if all(nm in names for nm in _FP_ARMS):
            for nm in _FP_ARMS: RT[f"{nm}/production"] = ratio(aligned[nm], prod)
        R[f"ratios_{tag}"] = RT
        # paired per-iteration region-wall delta (Codex Amdahl): dispatch/preproc/combine are byte-identical
        # between production and compute_swap, so the region-wall delta IS the compute-swap delta.
        # delta_us > 0  <=>  compute_swap is FASTER than production by that many us on that iteration.
        if "frozen_n2r" in names:
            dur = prod - aligned["frozen_n2r"]   # >0 => the folded region is FASTER than production
            R[f"delta_n2r_us_{tag}"] = dict(p5=p(dur, 5), p50=p(dur, 50), p95=p(dur, 95), mean=float(np.mean(dur)),
                                            n_faster=int((dur > 0).sum()), n=int(dur.size))
            if "frozen_n2" in names:
                dnr = aligned["frozen_n2"] - aligned["frozen_n2r"]   # >0 => the fold is FASTER than standalone rendezvous
                R[f"delta_n2r_vs_n2_us_{tag}"] = dict(p5=p(dnr, 5), p50=p(dnr, 50), p95=p(dnr, 95), mean=float(np.mean(dnr)),
                                                      n_faster=int((dnr > 0).sum()), n=int(dnr.size))
        if "compute_swap_n2" in names:
            dun = prod - aligned["compute_swap_n2"]   # >0 => n2 FASTER than production (the headline)
            R[f"delta_n2_us_{tag}"] = dict(p5=p(dun, 5), p50=p(dun, 50), p95=p(dun, 95), mean=float(np.mean(dun)),
                                           n_faster=int((dun > 0).sum()), n=int(dun.size))
            if "compute_swap" in names:
                dno = aligned["compute_swap"] - aligned["compute_swap_n2"]   # >0 => n2 FASTER than oracle n1g
                R[f"delta_n2_vs_oracle_us_{tag}"] = dict(p5=p(dno, 5), p50=p(dno, 50), p95=p(dno, 95), mean=float(np.mean(dno)),
                                                         n_faster=int((dno > 0).sum()), n=int(dno.size))
        if "compute_swap" in names:
            dus = prod - aligned["compute_swap"]     # >0 => compute_swap faster; report p5/p50/p95 (p95=optimistic saving)
            R[f"delta_us_{tag}"] = dict(p5=p(dus, 5), p50=p(dus, 50), p95=p(dus, 95), mean=float(np.mean(dus)),
                                        n_faster=int((dus > 0).sum()), n=int(dus.size))
        if "k0d_sq_b2f" in names and "k0d_sq" in names:
            db2f = aligned["k0d_sq"] - aligned["k0d_sq_b2f"]
            R[f"delta_k0d_sq_b2f_vs_sq_us_{tag}"] = dict(
                p5=p(db2f, 5), p50=p(db2f, 50), p95=p(db2f, 95),
                mean=float(np.mean(db2f)), n_faster=int((db2f > 0).sum()),
                n=int(db2f.size),
            )
        if "compute_swap_c32" in names:
            dusc = prod - aligned["compute_swap_c32"]   # >0 => c32 FASTER than production (closes the gap)
            R[f"delta_c32_us_{tag}"] = dict(p5=p(dusc, 5), p50=p(dusc, 50), p95=p(dusc, 95), mean=float(np.mean(dusc)),
                                            n_faster=int((dusc > 0).sum()), n=int(dusc.size))
            if "compute_swap" in names:
                dco = aligned["compute_swap"] - aligned["compute_swap_c32"]   # >0 => c32 FASTER than oracle n1g (the tail-aware win)
                R[f"delta_c32_vs_oracle_us_{tag}"] = dict(p5=p(dco, 5), p50=p(dco, 50), p95=p(dco, 95), mean=float(np.mean(dco)),
                                                          n_faster=int((dco > 0).sum()), n=int(dco.size))
        if "compute_swap_ah" in names:
            dusa = prod - aligned["compute_swap_ah"]     # >0 => ah FASTER than production (closes the gap)
            R[f"delta_ah_us_{tag}"] = dict(p5=p(dusa, 5), p50=p(dusa, 50), p95=p(dusa, 95), mean=float(np.mean(dusa)),
                                           n_faster=int((dusa > 0).sum()), n=int(dusa.size))
            if "compute_swap" in names:
                dao = aligned["compute_swap"] - aligned["compute_swap_ah"]   # >0 => ah FASTER than oracle n1g (the address-hoist win)
                R[f"delta_ah_vs_oracle_us_{tag}"] = dict(p5=p(dao, 5), p50=p(dao, 50), p95=p(dao, 95), mean=float(np.mean(dao)),
                                                         n_faster=int((dao > 0).sum()), n=int(dao.size))
        if "compute_swap_ecs" in names:
            duse = prod - aligned["compute_swap_ecs"]    # >0 => ecs FASTER than production (closes the gap)
            R[f"delta_ecs_us_{tag}"] = dict(p5=p(duse, 5), p50=p(duse, 50), p95=p(duse, 95), mean=float(np.mean(duse)),
                                            n_faster=int((duse > 0).sum()), n=int(duse.size))
            if "compute_swap" in names:
                deo = aligned["compute_swap"] - aligned["compute_swap_ecs"]  # >0 => ecs FASTER than oracle n1g (the co-scheduling win)
                R[f"delta_ecs_vs_oracle_us_{tag}"] = dict(p5=p(deo, 5), p50=p(deo, 50), p95=p(deo, 95), mean=float(np.mean(deo)),
                                                          n_faster=int((deo > 0).sum()), n=int(deo.size))
        if "compute_swap_ecs_d2" in names:
            dud2 = prod - aligned["compute_swap_ecs_d2"]   # >0 => ecs_d2 FASTER than production (closes the gap)
            R[f"delta_ecs_d2_us_{tag}"] = dict(p5=p(dud2, 5), p50=p(dud2, 50), p95=p(dud2, 95), mean=float(np.mean(dud2)),
                                               n_faster=int((dud2 > 0).sum()), n=int(dud2.size))
            if "compute_swap" in names:
                dd2o = aligned["compute_swap"] - aligned["compute_swap_ecs_d2"]  # >0 => ecs_d2 FASTER than oracle n1g (the D=2 win)
                R[f"delta_ecs_d2_vs_oracle_us_{tag}"] = dict(p5=p(dd2o, 5), p50=p(dd2o, 50), p95=p(dd2o, 95), mean=float(np.mean(dd2o)),
                                                             n_faster=int((dd2o > 0).sum()), n=int(dd2o.size))
        # ---- WEIGHT-FOOTPRINT 2x2 verdict (Codex 019f8db1) — aligned-MAX per-iteration deltas ----
        if all(nm in names for nm in _FP_ARMS):
            def _st(a): return dict(p5=p(a, 5), p50=p(a, 50), p95=p(a, 95), mean=float(np.mean(a)))
            n1g_delta   = aligned["fp_n1g_real"]   - aligned["fp_n1g_alias"]     # >0 => real slower: n1g's weight-DRAM cost
            aiter_delta = aligned["fp_aiter_real"] - aligned["fp_aiter_alias"]   # aiter core's weight-DRAM cost
            real_gap    = aligned["fp_n1g_real"]   - aligned["fp_aiter_real"]    # the real n1g-vs-aiter gap (~38us sanity)
            alias_gap   = aligned["fp_n1g_alias"]  - aligned["fp_aiter_alias"]   # gap with weights cache-resident
            dod         = real_gap - alias_gap                                   # DoD: portion of the gap the weight stream UNIQUELY costs n1g
            R[f"fp_n1g_delta_us_{tag}"]   = _st(n1g_delta)
            R[f"fp_aiter_delta_us_{tag}"] = _st(aiter_delta)
            R[f"fp_real_gap_us_{tag}"]    = _st(real_gap)
            R[f"fp_alias_gap_us_{tag}"]   = _st(alias_gap)
            R[f"fp_DoD_us_{tag}"]         = _st(dod)
            R[f"fp_cosched_headroom_est_us_{tag}"] = 0.246 * p(n1g_delta, 50)    # §L22 reread fraction 40.20/163.32; SIZING PRIOR only
        if rank == 0:
            print(f"[E002F {tag} TIMING us] {json.dumps(R[f'timing_us_{tag}'])}", flush=True)
            print(f"[E002F {tag} RATIOS] {json.dumps(R[f'ratios_{tag}'])}", flush=True)
            if "compute_swap_n2" in names: print(f"[E002F {tag} N2 DELTA us] prod-n2={json.dumps(R[f'delta_n2_us_{tag}'])}"
                  + (f" oracle-n2={json.dumps(R[f'delta_n2_vs_oracle_us_{tag}'])}" if 'compute_swap' in names else ""), flush=True)
            if "frozen_n2r" in names: print(f"[E002F {tag} N2R DELTA us] prod-n2r={json.dumps(R[f'delta_n2r_us_{tag}'])}", flush=True)
            if "compute_swap" in names: print(f"[E002F {tag} DELTA us] {json.dumps(R[f'delta_us_{tag}'])}", flush=True)
            if "compute_swap_c32" in names: print(f"[E002F {tag} C32 DELTA us] prod-c32={json.dumps(R[f'delta_c32_us_{tag}'])}"
                  + (f" oracle-c32={json.dumps(R[f'delta_c32_vs_oracle_us_{tag}'])}" if 'compute_swap' in names else ""), flush=True)
            if "compute_swap_ah" in names: print(f"[E002F {tag} AH DELTA us] prod-ah={json.dumps(R[f'delta_ah_us_{tag}'])}"
                  + (f" oracle-ah={json.dumps(R[f'delta_ah_vs_oracle_us_{tag}'])}" if 'compute_swap' in names else ""), flush=True)
            if "compute_swap_ecs" in names: print(f"[E002F {tag} ECS DELTA us] prod-ecs={json.dumps(R[f'delta_ecs_us_{tag}'])}"
                  + (f" oracle-ecs={json.dumps(R[f'delta_ecs_vs_oracle_us_{tag}'])}" if 'compute_swap' in names else ""), flush=True)
            if "compute_swap_ecs_d2" in names: print(f"[E002F {tag} ECS-D2 DELTA us] prod-ecs_d2={json.dumps(R[f'delta_ecs_d2_us_{tag}'])}"
                  + (f" oracle-ecs_d2={json.dumps(R[f'delta_ecs_d2_vs_oracle_us_{tag}'])}" if 'compute_swap' in names else ""), flush=True)
            if all(nm in names for nm in _FP_ARMS):
                print(f"[E002F {tag} FOOTPRINT us] n1g_delta(real-alias)={json.dumps(R[f'fp_n1g_delta_us_{tag}'])} "
                      f"aiter_delta={json.dumps(R[f'fp_aiter_delta_us_{tag}'])}", flush=True)
                print(f"[E002F {tag} FOOTPRINT us] real_gap(n1g-aiter)={json.dumps(R[f'fp_real_gap_us_{tag}'])} "
                      f"alias_gap={json.dumps(R[f'fp_alias_gap_us_{tag}'])} DoD={json.dumps(R[f'fp_DoD_us_{tag}'])} "
                      f"cosched_headroom_est_us={R[f'fp_cosched_headroom_est_us_{tag}']:.2f}", flush=True)

    summarize(run_condition(False, "warm"), "warm")   # canonical warm interleaved AB/BA
    summarize(run_condition(True,  "cold"), "cold")    # cache-conditioned interleaved AB/BA (do NOT pool with warm)

R["config"] = dict(
    incl_plan=INCL_PLAN, T=T, t_loc_max=T_LOC_MAX, padmax=PADMAX,
    block_num=BLK, warp_num=WARP, gblk=GBLK, cblk=CBLK,
    scrub_mb=R.get("scrub_mb"), ntimed=NTIMED, nuntimed=NUNTIMED,
    arms=R["arms"], corpus=REGION_CORPUS, maxtok=MAXTOK,
    maxtok_prod=MAXTOK_PROD, pf_hsaco_dir=PF_HSACO_DIR,
    input_mode=K0_INPUT_MODE, benchmark_protocol=K0_BENCHMARK_PROTOCOL,
    synthetic_route=K0_SYNTH_ROUTE or None, synthetic_seed=K0_SYNTH_SEED,
    synthetic_weight_mode=K0_SYNTH_WEIGHT_MODE,
    pf6_grid=(
        dict(ctas=PF6_MEGA_GRID, threads=PF6_THREADS)
        if PF6_REQUESTED else None
    ),
)
hbarrier()
OUTDIR = os.environ.get("K0_OUT_DIR", OUT); os.makedirs(OUTDIR, exist_ok=True)
OUTJSON = f"{OUTDIR}/k0pf_prefill_rank{rank}.json"
with open(OUTJSON, "w") as f: json.dump(R, f, indent=2)
if rank == 0:
    print(f"[E002F r{rank}] T_loc={T_loc} cap_ok={cap_ok} gate_ok={gate_ok} "
          f"ctrl_fails={R.get('control_fails')} bridge_ctrl_fails={R.get('bridge_control_fails')} "
          f"oracle_pass={R.get('bridge_oracle',{}).get('pass')} "
          f"oracle_vs_prod_relL2={R.get('oracle_vs_prod',{}).get('rel_L2')} "
          f"inner_n1g_vs_fmoe_relL2={R.get('inner_n1g_vs_fmoe',{}).get('rel_L2')}", flush=True)
    print(f"[E002F SUMMARY warm] ratios={json.dumps(R.get('ratios_warm',{}))} delta_us={json.dumps(R.get('delta_us_warm',{}))}", flush=True)
    print(f"[E002F SUMMARY cold] ratios={json.dumps(R.get('ratios_cold',{}))} delta_us={json.dumps(R.get('delta_us_cold',{}))}", flush=True)
hbarrier()
try: dist.destroy_process_group()
except Exception: pass
