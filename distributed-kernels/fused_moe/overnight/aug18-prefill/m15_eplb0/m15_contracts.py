"""Frozen M15 fused-MoE megakernel contracts (additive to the PF4H packet).

Nothing in this module imports torch, vLLM, ROCm, or MoRI.  It is the CPU-side
authority for:

* the 63-word ``K0P6_D_*`` descriptor dictionary mirrored from
  ``distributed-kernels/fused_moe/k0pf6gm_device_tile_m15.hip`` (slots 0..54
  plus ``K0P6_D_SYMMETRIC``) and
  ``distributed-kernels/fused_moe/moe_mps_adapter.cuh`` (slots 56..62);
* a pure-Python, bit-identical mirror of
  ``hk_moe::mps::encode_config`` / ``decode_config`` / ``config_is_valid``;
* the exact buffer shape table for the only validated serving shape
  (balanced ``T=4096`` on eight ranks); and
* the host-side attestation of every device entry-guard condition, so a bad
  build refuses *before* graph capture rather than through the device-side
  ``pperr`` fail-closed path (which fires after capture and is therefore
  unrecoverable for a captured graph).

The device is the authority for semantics; this module is the authority for
"the host built what the device demands".  Every constant here is duplicated
from a named source line so the structural tests can diff the two.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .contracts import FROZEN_CONTRACT
from .errors import ActivationBlocker, PF4HActivationError


# --------------------------------------------------------------------------
# Opt-in
# --------------------------------------------------------------------------

ENV_MODE = "VLLM_PF4H_INTEGRATION_MODE"
M15_MODE_VALUE = "m15"
RING_SLOTS_ENV = "VLLM_PF4H_M15_RING_SLOTS"
ALLOCATE_PART_ENV = "VLLM_PF4H_M15_ALLOCATE_PART"
EPLB_COMPOSE_ENV = "VLLM_PF4H_M15_EPLB"

# The M15 source revision this packet is pinned against.  Bump together with
# ``M15_SOURCE_SHA256`` in ``m15_sources.py`` whenever the .hip changes.
M15_SRC_REV = 1

#: Exported by the kernel; the packet refuses any other symbol.
M15_KERNEL_SYMBOL = "k0pf6gm_m15_mega"
#: One-thread helper that snapshots MoRI's ``GpuStates`` heap bases.
M15_SNAPSHOT_SYMBOL = "k0pf6_mori_heap_snapshot"

#: Compile-time arms this packet requires to be OFF.  Any other value changes
#: the descriptor length (STAGED adds slot 63) or the scatter plan.
M15_REQUIRED_DEFINES: Mapping[str, int] = {
    "K0P6_M15_STAGED": 0,
    "K0P6_M15_SCATTER_RR": 0,
    "K0P6_MPS_ENABLE_MODE14": 0,
    "K0P6_MPS_ENABLE_TBO": 0,
    "K0P6_MPS_E23_RING": 0,
    # exp_27 shipped arm; M6 reads token-major sc_stage and M5's transpose is
    # dead.  ``part_is_dead`` depends on this being 1.
    "K0P6_MPS_ASCALE_TM": 1,
    "K0P6GM_G": 3,
}


def m15_mode_selected(environ: Mapping[str, str] | None = None) -> bool:
    """True only for the exact string ``m15``; aliases never activate."""

    return (os.environ if environ is None else environ).get(
        ENV_MODE
    ) == M15_MODE_VALUE


# --------------------------------------------------------------------------
# Descriptor dictionary — mirrors the .hip's own offset block verbatim
# --------------------------------------------------------------------------


class DescriptorSlot(int, Enum):
    """``K0P6_D_*`` from k0pf6gm_device_tile_m15.hip lines 100..155."""

    MYIDS = 0
    MYWG = 1
    HIDDEN = 2
    A_LL = 3
    DESTCTR = 4
    ROWS_DONE = 5
    PUSHED = 6
    QPUSH_DONE = 7
    A_DST = 8
    SC_STAGE = 9
    RECV_EID = 10
    RECV_WGT = 11
    PULL_STAGE = 12
    PULL_CNT = 13
    STI = 14
    SWT = 15
    SEI = 16
    NVI = 17
    PULL_PTR = 18
    PULL_SRC = 19
    SC_DST = 20
    PART = 21
    SCRATCH = 22
    GBAR = 23
    HCNT = 24
    ROW_READY = 25
    ROW_REM = 26
    A2Q = 27
    DQ2 = 28
    A2_DONE = 29
    PART_DONE = 30
    RETIRED = 31
    OUT = 32
    COMB_DONE = 33
    EPOCH_CELL = 34
    W13 = 35
    S13 = 36
    W2 = 37
    S2 = 38
    CUR = 39
    WORLD = 40
    SPIN = 41
    PPERR = 42
    MAXB = 43
    T = 44
    TOPK = 45
    E = 46
    TLOCMAX = 47
    PADMAX = 48
    DEBUG = 49
    CHUNK_READY = 50
    MAXTOK = 51
    SPIN_DBG = 52
    TILE_DESC = 53
    NUM_TILES = 54
    SYMMETRIC = 55
    # moe_mps_adapter.cuh lines 20..27
    MPS_Q = 56
    MPS_NCARR = 57
    MPS_PUSHED = 58
    MPS_CLAIM = 59
    MPS_STATE = 60
    MPS_SLOTS = 61
    MPS_CFG = 62


#: ``K0P6_MPS_D_LEN``; slot 63 exists only under ``K0P6_M15_STAGED=1``.
M15_DESCRIPTOR_WORDS = 63

#: Slots holding a plain integer value rather than a device address.
SCALAR_SLOTS = frozenset(
    {
        DescriptorSlot.CUR,
        DescriptorSlot.WORLD,
        DescriptorSlot.SPIN,
        DescriptorSlot.MAXB,
        DescriptorSlot.T,
        DescriptorSlot.TOPK,
        DescriptorSlot.E,
        DescriptorSlot.TLOCMAX,
        DescriptorSlot.PADMAX,
        DescriptorSlot.DEBUG,
        DescriptorSlot.MAXTOK,
        DescriptorSlot.MPS_CFG,
    }
)

#: Slots that must live in the MoRI symmetric heap because the kernel reaches
#: them on a peer through ``hk_moe::peer_ptr`` (heap-relative translation).
#: Mirrors the ``mori_t`` allocations in
#: ``prefill_opt/host/e004pf_k0pf_ab.py`` lines 1464..1520.
SYMMETRIC_SLOTS = (
    DescriptorSlot.A_LL,
    DescriptorSlot.DESTCTR,
    DescriptorSlot.ROWS_DONE,
    DescriptorSlot.PART,
    DescriptorSlot.ROW_READY,
    DescriptorSlot.RETIRED,
    DescriptorSlot.CHUNK_READY,
    DescriptorSlot.MPS_SLOTS,
)

#: Slots whose value is a live vLLM allocation that only exists per call.
#: These are the four the descriptor cannot own; see ``m15_runtime`` for the
#: capture-time latch protocol.
DYNAMIC_SLOTS = (
    DescriptorSlot.MYIDS,
    DescriptorSlot.MYWG,
    DescriptorSlot.HIDDEN,
    DescriptorSlot.OUT,
)

#: Slots bound once per routed layer from the stock shuffled AITER weights.
WEIGHT_SLOTS = (
    DescriptorSlot.W13,
    DescriptorSlot.S13,
    DescriptorSlot.W2,
    DescriptorSlot.S2,
)


# --------------------------------------------------------------------------
# Packed MPS config word — pure-Python mirror of moe_mps_adapter.cuh
# --------------------------------------------------------------------------

_U64 = (1 << 64) - 1

K_MODE_REMOTE_ACCUM = 12
K_MODE_DIRECT_ROWS = 13
K_REMOTE_ACCUM_G_MASK = 0x0F
K_REMOTE_ACCUM_DETECT_BIT = 0x10
K_REMOTE_ACCUM_THROTTLE_BIT = 0x20
K_REMOTE_ACCUM_SKIP_PART_ZERO_BIT = 0x40
K_COARSE_KEEP_DRAIN_BIT = 0x80
K_REMOTE_ACCUM_THROTTLE_DEPTH_MASK = 0x300
K_REMOTE_ACCUM_THROTTLE_DEPTH_SHIFT = 8
K_CTAS_PER_XCD = 32

# With MODE14/TBO compiled out (M15_REQUIRED_DEFINES) the legal g bits are
# exactly these; kCoarseKeepDrainBit is excluded by the #if.
K_REMOTE_ACCUM_G_LEGAL_BITS = (
    K_REMOTE_ACCUM_G_MASK
    | K_REMOTE_ACCUM_DETECT_BIT
    | K_REMOTE_ACCUM_THROTTLE_BIT
    | K_REMOTE_ACCUM_SKIP_PART_ZERO_BIT
    | K_REMOTE_ACCUM_THROTTLE_DEPTH_MASK
)


@dataclass(frozen=True)
class MpsConfig:
    """``hk_moe::mps::config``."""

    reserved_comm_ctas: int
    group_slices: int
    mode: int
    flush_rows: int
    pull_fallback: bool = False
    timestamps: bool = False


def encode_config(config: MpsConfig) -> int:
    """Bit-identical mirror of ``hk_moe::mps::encode_config``."""

    return (
        (config.reserved_comm_ctas & _U64)
        | ((config.group_slices & 0xFF) << 8)
        | ((config.mode & _U64) << 16)
        | ((config.flush_rows & _U64) << 24)
        | ((1 if config.pull_fallback else 0) << 32)
        | ((1 if config.timestamps else 0) << 33)
        | (((config.group_slices >> 8) & 0xFF) << 34)
    ) & _U64


def decode_config(word: int) -> MpsConfig:
    """Bit-identical mirror of ``hk_moe::mps::decode_config``."""

    word &= _U64
    return MpsConfig(
        reserved_comm_ctas=word & 0xFF,
        group_slices=((word >> 8) & 0xFF) | (((word >> 34) & 0xFF) << 8),
        mode=(word >> 16) & 0xFF,
        flush_rows=(word >> 24) & 0xFF,
        pull_fallback=bool((word >> 32) & 1),
        timestamps=bool((word >> 33) & 1),
    )


def mode_is_direct_accum(config: MpsConfig) -> bool:
    """``mode_is_direct_accum`` with MODE14 and TBO compiled out."""

    return config.mode in (K_MODE_REMOTE_ACCUM, K_MODE_DIRECT_ROWS)


def mode_is_stream(config: MpsConfig) -> bool:
    return config.mode == 2 or config.mode in (7, 8, 9, 10, 11)


def _mode_is_diag_pool(config: MpsConfig) -> bool:
    return config.mode in (5, 6)


def config_is_valid(config: MpsConfig) -> bool:
    """``hk_moe::mps::config_is_valid`` for the default (flags-off) build."""

    if config.reserved_comm_ctas > 128:
        return False
    if config.reserved_comm_ctas >= 256:
        return False
    if mode_is_direct_accum(config):
        if (config.group_slices & K_REMOTE_ACCUM_G_MASK) != 1:
            return False
        if (config.group_slices & ~K_REMOTE_ACCUM_G_LEGAL_BITS) != 0:
            return False
        if config.pull_fallback:
            return False
        if (
            config.group_slices & K_REMOTE_ACCUM_THROTTLE_DEPTH_MASK
        ) != 0 and (config.group_slices & K_REMOTE_ACCUM_THROTTLE_BIT) == 0:
            return False
        if (config.group_slices & K_REMOTE_ACCUM_SKIP_PART_ZERO_BIT) != 0:
            if config.mode != K_MODE_REMOTE_ACCUM:
                return False
            if (config.group_slices & K_REMOTE_ACCUM_DETECT_BIT) != 0:
                return False
    elif config.group_slices not in (1, 2, 4, 16):
        return False
    if config.mode > 13:
        return False
    if config.mode == 7 and not config.pull_fallback:
        return False
    if mode_is_stream(config) and config.reserved_comm_ctas == 0:
        return False
    if _mode_is_diag_pool(config) and config.reserved_comm_ctas == 0:
        return False
    if config.mode == 5 and config.group_slices == 16:
        return False
    if config.mode == 3 and (config.reserved_comm_ctas % K_CTAS_PER_XCD) != 0:
        return False
    if config.mode == 1 and config.reserved_comm_ctas != 0:
        return False
    if config.flush_rows == 0 or config.flush_rows > 64:
        return False
    return True


def skip_dead_part_zero(config: MpsConfig) -> bool:
    """True when M5's ~448 MiB ``part`` zero-fill is compiled away (exp_24 A)."""

    return (
        config.mode == K_MODE_REMOTE_ACCUM
        and (config.group_slices & K_REMOTE_ACCUM_SKIP_PART_ZERO_BIT) != 0
        and (config.group_slices & K_REMOTE_ACCUM_DETECT_BIT) == 0
    )


def part_is_dead(config: MpsConfig) -> bool:
    """True when descriptor slot 21 (``part``) is never *dereferenced*.

    This is a line-cited proof over the exact sources this packet pins, not a
    citation of exp_24's prose.  Every site that touches ``part`` in the M15
    translation unit:

    * ``k0pf6gm_device_tile_m15.hip:937`` -- reads the descriptor word into a
      pointer variable.  No dereference; a null word is legal here.
    * ``:988`` -- M5's ``zero_part_scale_transpose``, guarded by
      ``if (!skip_dead_part_zero(cfg4))`` under ``K0P6_MPS_ASCALE_TM``.
      Skipped whenever :func:`skip_dead_part_zero` holds.
    * ``:1002`` -- the same call in the ``#else`` arm of
      ``K0P6_MPS_ASCALE_TM``; not compiled when ASCALE_TM is 1.
    * ``:1097`` -- passes the word to ``n2p6m15_phase2_body`` as ``OUT``.

    And inside ``n2_phase2_gm_mps.cpp`` the parameter ``OUT`` is dereferenced
    at exactly two places:

    * ``:243`` -- inside ``if (dual)``, the lost-update detector, which is the
      ``kRemoteAccumDetectBit`` (0x10).  ``config_is_valid`` *rejects* that bit
      together with the skip-part-zero bit, so it cannot be set here.
    * ``:256`` -- the ``peer_tab == nullptr`` arm, i.e. the non
      direct-accumulate transports.  Mode 12 sets ``m7_peer_tab = m7tab``
      (``:391``), so this arm is not taken.

    Therefore, for mode 12 with the skip bit set, the detect bit clear and
    ``K0P6_MPS_ASCALE_TM == 1``, slot 21 is read as a word and never
    dereferenced.  Any other configuration returns False and the caller must
    allocate the full 560 MiB.
    """

    return (
        skip_dead_part_zero(config)
        and M15_REQUIRED_DEFINES["K0P6_MPS_ASCALE_TM"] == 1
    )


def allocate_part(
    config: MpsConfig,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Whether to spend 560 MiB per ring slot on the ``part`` tower.

    Fail-closed by construction: the allocation is elided *only* when
    :func:`part_is_dead` proves it is never dereferenced for this exact
    config.  A config that does not satisfy the proof allocates, regardless of
    the environment.  ``VLLM_PF4H_M15_ALLOCATE_PART=1`` forces allocation back
    on for a config that would otherwise elide it.
    """

    forced = (os.environ if environ is None else environ).get(
        ALLOCATE_PART_ENV
    )
    if forced is not None and forced not in ("0", "1"):
        raise PF4HActivationError(
            (
                ActivationBlocker(
                    "M15-ENV-003",
                    f"{ALLOCATE_PART_ENV}={forced!r} is not exactly '0' or '1'",
                    "unset it to follow the deadness proof",
                ),
            )
        )
    if forced == "1":
        return True
    return not part_is_dead(config)


def eplb_compose_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """OFF by default; ``1`` composes M15 with vLLM EPLB placement-only.

    With the variable unset the PF4H-FULL-018 blocker stands verbatim and
    ``enable_eplb`` refuses activation, so the default build's behaviour is
    byte-identical.  ``1`` swaps that single blocker for the compose gate in
    :func:`m15_eplb.validate_eplb_compose`, which demands zero redundant
    experts, synchronous EPLB, no elastic EP, and no M18/M20 replica pools.

    Nothing on the data path changes: vLLM already resolves logical expert
    ids to physical slot ids inside the router
    (``BaseRouter._select_experts`` step 3), so the topk_ids the megakernel
    receives are physical and ``dest = eid / E`` is already correct.
    """

    raw = (os.environ if environ is None else environ).get(EPLB_COMPOSE_ENV)
    if raw is None:
        return False
    if raw not in ("0", "1"):
        raise PF4HActivationError(
            (
                ActivationBlocker(
                    "M15-ENV-004",
                    f"{EPLB_COMPOSE_ENV}={raw!r} is not exactly '0' or '1'",
                    "unset it to keep EPLB refused",
                ),
            )
        )
    return raw == "1"


#: The serving arm: exp_03/aug13's mode-12 ratchet at the M15 C-response
#: optimum.  C=28 service CTAs, g=353 (physical g 1 | throttle | skip dead
#: part zero | depth selector 01 -> vmcnt(4)), flush_rows=16.
M15_SERVING_CONFIG = MpsConfig(
    reserved_comm_ctas=28,
    group_slices=353,
    mode=K_MODE_REMOTE_ACCUM,
    flush_rows=16,
)


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------

#: ``K0P5_ROW_STRIDE`` from prefill_opt/kernels/k0pf5_ll128.hpp:33.
K0P5_ROW_STRIDE = 8064
#: ``K0P6_CHUNK`` from prefill_opt/kernels/k0pf6_chunk_release.hpp:35.
K0P6_CHUNK = 512
#: ``K0P6C_NCHUNK_MAX`` from k0pf6gm_device_tile_m15.hip:90.
K0P6C_NCHUNK_MAX = 8
#: ``K0P6_MAXE`` from k0pf6gm_device_tile_m15.hip:92.
K0P6_MAXE = 64
#: ``K0P6_M15_SLABS`` from k0pf6gm_device_tile_m15.hip:177.
K0P6_M15_SLABS = 2
#: ``K0P6_MPS_ST_WORDS + 2 * K0P6_MPS_TS_COUNT`` uint32 words zeroed by M0.
M15_MPS_STATE_U32_WORDS = 8 + 2 * 8
#: ``K0P4_MAXE * K0P4_MAXCTA`` histogram table.
M15_HCNT_WORDS = 64 * 256
#: Threads and CTAs are frozen: the entry guard demands ``gridDim.x == 256``.
M15_GRID = 256
M15_BLOCK = 256
#: Device spin bound; identical value to the PF4H four-kernel path.
M15_SPIN_LIMIT = 20_000_000


@dataclass(frozen=True)
class M15GraphBucket:
    """One exact capture/replay shape for the megakernel.

    ``tokens_per_rank`` is a launch constant baked into slot ``T``.  A bucket
    captured for one instance is never a fallback for another; there is no
    nearest-size lookup anywhere in this packet.
    """

    tokens_per_rank: int
    maxtok: int
    t_loc_max: int
    pad_max: int
    world: int = FROZEN_CONTRACT.ep_size
    topk: int = FROZEN_CONTRACT.topk
    local_experts: int = FROZEN_CONTRACT.local_experts
    hidden_size: int = FROZEN_CONTRACT.hidden_size
    quant_block: int = FROZEN_CONTRACT.quant_block

    # -- derived ---------------------------------------------------------
    @property
    def t_ext(self) -> int:
        """``world * MAXTOK``; the receive-row space M15 indexes by segment."""

        return self.world * self.maxtok

    @property
    def maxb(self) -> int:
        """``K0P6_D_MAXB``: number of 32-row expert blocks."""

        return self.pad_max // 32

    @property
    def num_groups(self) -> int:
        """Per-row FP8 scale groups (``NG``)."""

        return self.hidden_size // self.quant_block

    @property
    def n2_rowcap(self) -> int:
        """N2's BM32 handoff capacity; deliberately larger than ``pad_max``."""

        return (
            self.world * self.tokens_per_rank * self.topk
            + FROZEN_CONTRACT.global_experts * 32
            - self.topk
        )

    @property
    def chunks_per_segment(self) -> int:
        return (self.maxtok + K0P6_CHUNK - 1) // K0P6_CHUNK

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.tokens_per_rank, self.maxtok, self.t_loc_max, self.pad_max)


#: The only shape M15 is validated at (balanced exact-B4096 on eight ranks).
M15_PREFILL_B4096_BUCKET = M15GraphBucket(
    tokens_per_rank=4096,
    maxtok=4096,
    t_loc_max=40_960,
    pad_max=263_136,
)


@dataclass(frozen=True)
class BufferSpec:
    """One descriptor-bound allocation.

    ``elements`` is the element count, ``itemsize`` the element width in
    bytes, ``symmetric`` whether it must come from the MoRI heap, and
    ``zero`` whether the host must zero it before the first epoch.
    """

    slot: DescriptorSlot
    name: str
    elements: int
    itemsize: int
    dtype: str
    symmetric: bool
    zero: bool
    note: str = ""

    @property
    def nbytes(self) -> int:
        return self.elements * self.itemsize


def buffer_table(
    bucket: M15GraphBucket,
    *,
    elide_part: bool = False,
    adaptive: bool = False,
    m20: bool = False,
) -> tuple[BufferSpec, ...]:
    """Every descriptor-bound allocation, in a rank-deterministic order.

    ``elide_part`` drops the 560 MiB ``part`` tower, which
    :func:`part_is_dead` proves is never dereferenced at mode 12 with the skip
    bit set.  Slot 21 then carries a null word -- the same value the kernel
    passes internally on that branch -- so a violated proof faults loudly
    instead of silently corrupting a live buffer.  Eliding also removes
    ``part`` from the symmetric block, which keeps the remaining offsets
    contiguous and identical on every rank.

    The symmetric entries are emitted first and in a fixed order because M15
    addresses peers through *heap-relative* translation
    (``hk_moe::peer_ptr`` -> ``kittens::distributed::translate_peer``): every
    rank must obtain the same offset from its own heap base for the same
    logical buffer.  Any divergence in allocation order across ranks silently
    corrupts every remote write.  ``m15_runtime`` additionally *proves* the
    offsets match with a collective before capture.
    """

    b = bucket
    c = FROZEN_CONTRACT
    ng = b.num_groups
    part_spec: tuple[BufferSpec, ...] = (
        ()
        if elide_part
        else (
            BufferSpec(
                DescriptorSlot.PART,
                "part",
                b.t_loc_max * b.hidden_size,
                2,
                "bfloat16",
                True,
                True,
                "dead under mode 12 + skip-part-zero; see part_is_dead()",
            ),
        )
    )
    return (
        # ---- symmetric (MoRI heap; order is part of the ABI) ------------
        BufferSpec(
            DescriptorSlot.A_LL,
            "a_ll",
            b.t_loc_max * K0P5_ROW_STRIDE,
            1,
            "uint8",
            True,
            True,
            "LL128 dispatch landing, 63*128 bytes per row",
        ),
        BufferSpec(
            DescriptorSlot.ROWS_DONE,
            "rows_done",
            b.world,
            8,
            "int64",
            True,
            True,
            "per-source row completion counters",
        ),
        BufferSpec(
            DescriptorSlot.DESTCTR,
            "dest_counter",
            2 * b.world,
            4,
            "int32",
            True,
            True,
            "epoch-parity double buffered [2][world]",
        ),
        BufferSpec(
            DescriptorSlot.CHUNK_READY,
            "chunk_ready",
            b.world * K0P6C_NCHUNK_MAX,
            8,
            "int64",
            True,
            True,
            "self-describing (epoch<<32 | fill) chunk release words",
        ),
        *part_spec,
        BufferSpec(
            DescriptorSlot.ROW_READY,
            "row_ready",
            b.world * b.t_loc_max,
            4,
            "int32",
            True,
            True,
            "per-producer receive-row flags; M15 parks two slab words at "
            "T_ext and T_ext+1 of each segment",
        ),
        BufferSpec(
            DescriptorSlot.RETIRED,
            "retired",
            b.world,
            8,
            "int64",
            True,
            True,
            "M0 retire-wait / M9 retire-poke",
        ),
        BufferSpec(
            DescriptorSlot.MPS_SLOTS,
            "mps_slots",
            b.world * b.maxtok * b.hidden_size,
            2,
            "bfloat16",
            True,
            True,
            "owner-resident landing slots; 16-byte aligned (packet16)",
        ),
        # M19 only, appended AFTER every historical symmetric entry so the
        # non-adaptive arms' attested carve offsets are untouched.  Slot 64
        # is outside the DescriptorSlot enum (adaptive descriptors are 65
        # words); u32[2 parity][world][8] peer-written decision words.
        *(
            (
                BufferSpec(
                    64,  # K0P6_D_M15_REPDEC
                    "rep_dec",
                    2 * b.world * 8,
                    4,
                    "int32",
                    True,
                    True,
                    "per-source per-chunk replication decisions (M19)",
                ),
            )
            if adaptive
            else ()
        ),
        # ---- local ------------------------------------------------------
        BufferSpec(
            DescriptorSlot.A_DST,
            "a_dst",
            b.t_loc_max * b.hidden_size,
            1,
            "uint8",
            False,
            True,
            "unpacked FP8 expert input rows",
        ),
        BufferSpec(
            DescriptorSlot.SC_STAGE,
            "sc_stage",
            b.t_loc_max * ng,
            4,
            "float32",
            False,
            True,
            "token-major arrival scales (K0P6_MPS_ASCALE_TM=1)",
        ),
        BufferSpec(
            DescriptorSlot.RECV_EID,
            "recv_eid",
            b.t_loc_max * b.topk,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.RECV_WGT,
            "recv_wgt",
            b.t_loc_max * b.topk,
            4,
            "float32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.PUSHED,
            "pushed_count",
            b.world,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.QPUSH_DONE,
            "qpush_done",
            1,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.PULL_STAGE,
            "pull_stage",
            b.tokens_per_rank * b.topk * 2,
            4,
            "int32",
            False,
            False,
            "host-initialized to -1",
        ),
        BufferSpec(
            DescriptorSlot.PULL_CNT,
            "pull_cnt",
            b.tokens_per_rank,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.STI, "sti", b.pad_max, 4, "int32", False, True
        ),
        BufferSpec(
            DescriptorSlot.SWT, "swt", b.pad_max, 4, "float32", False, True
        ),
        BufferSpec(
            DescriptorSlot.SEI, "sei", b.pad_max // 32, 4, "int32", False, True
        ),
        BufferSpec(DescriptorSlot.NVI, "nvi", 2, 4, "int32", False, True),
        BufferSpec(
            DescriptorSlot.PULL_PTR,
            "pull_ptr",
            b.tokens_per_rank + 1,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.PULL_SRC,
            "pull_src",
            b.tokens_per_rank * b.world * 2,
            4,
            "int32",
            False,
            False,
            "host-initialized to -1",
        ),
        BufferSpec(
            DescriptorSlot.SC_DST,
            "sc_dst",
            b.t_loc_max * ng,
            4,
            "float32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.SCRATCH, "scratch", 256, 4, "int32", False, True
        ),
        BufferSpec(DescriptorSlot.GBAR, "gbar", 1, 4, "int32", False, True),
        BufferSpec(
            DescriptorSlot.HCNT,
            "hcnt",
            M15_HCNT_WORDS,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.ROW_REM,
            "row_remaining",
            b.t_loc_max,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.A2Q,
            "a2q",
            b.n2_rowcap * 1024,
            2,
            "bfloat16",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.DQ2,
            "dq2",
            b.n2_rowcap * 16,
            4,
            "float32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.A2_DONE, "a2_done", b.maxb, 4, "int32", False, True
        ),
        BufferSpec(
            DescriptorSlot.PART_DONE,
            "part_done",
            b.maxb,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.COMB_DONE,
            "combine_done",
            1,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.EPOCH_CELL,
            "epoch_cell",
            1,
            4,
            "int32",
            False,
            True,
            "monotonic launch counter; NEVER host-reset between replays",
        ),
        BufferSpec(
            DescriptorSlot.PPERR,
            "pperr",
            1,
            4,
            "int32",
            False,
            True,
            "sticky device protocol error; NEVER host-reset between replays",
        ),
        BufferSpec(
            DescriptorSlot.SPIN_DBG, "spin_dbg", 2, 4, "int32", False, True
        ),
        BufferSpec(
            DescriptorSlot.TILE_DESC,
            "tile_desc",
            b.pad_max // 32,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.NUM_TILES,
            "num_tiles",
            2,
            4,
            "int32",
            False,
            True,
            "[0]=device tile count, [1]=compiled K0P6GM_G",
        ),
        BufferSpec(
            DescriptorSlot.MPS_Q,
            "mps_queue",
            (b.pad_max // 32) * 16,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.MPS_NCARR,
            "mps_arrivals",
            b.t_ext * 16,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.MPS_PUSHED,
            "mps_pushed",
            b.t_ext,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.MPS_CLAIM,
            "mps_claim",
            b.t_ext,
            4,
            "int32",
            False,
            True,
        ),
        BufferSpec(
            DescriptorSlot.MPS_STATE,
            "mps_state",
            M15_MPS_STATE_U32_WORDS // 2,
            8,
            "int64",
            False,
            True,
            "8 u32 scalars + 8 u64 timestamps = 96 bytes, 8-byte aligned",
        ),
        BufferSpec(
            DescriptorSlot.SYMMETRIC,
            "heap_descriptor",
            9,
            8,
            "int64",
            False,
            True,
            "MoRI GpuStates heapBaseAddr + 8 p2pPeerPtrs",
        ),
        # M20 only, appended LAST so every earlier (symmetric and local)
        # entry keeps its position.  Slot 65 is outside the DescriptorSlot
        # enum (m20 descriptors are 71 words); rank-LOCAL 8xu32 per-chunk
        # decision bitmap, rewritten on-stream by the prepare()-seam pre-op
        # before every cached-layer launch.  8-byte alignment (the device
        # entry guard's requirement) is guaranteed by the torch allocator.
        *(
            (
                BufferSpec(
                    65,  # K0P6_D_M20_DECIN
                    "decin",
                    8,
                    4,
                    "int32",
                    False,
                    True,
                    "precomputed per-chunk replication decisions (M20)",
                ),
            )
            if m20
            else ()
        ),
    )


def ring_slots(environ: Mapping[str, str] | None = None) -> int:
    """ONE slot by default; ``2`` is the documented opt-back-in.

    M15's own M0 retire-wait already blocks a rank from reusing a slot until
    every peer retired the previous epoch on that slot, so a single slot is
    protocol-safe on its own -- unlike PF4H's combine, which has no
    post-combine barrier and therefore *needs* the two-slot ring.

    The default is 1 because the second slot costs another ~1.6 GiB and the
    PF4H campaign already found that two resident B4096 graphs exhaust the
    MI350X headroom at ``--gpu-memory-utilization 0.70``.  A first gate that
    fails on memory would teach us nothing about the two assumptions that
    actually need testing (the capture-time pointer latch and heap-offset
    symmetry), so the budget is spent on those.  Whether the second slot buys
    measurable overlap of layer ``L+1``'s dispatch against layer ``L``'s tail
    is a server-side question to settle with numbers later.
    """

    raw = (os.environ if environ is None else environ).get(RING_SLOTS_ENV)
    if raw is None:
        return 1
    if raw not in ("1", "2"):
        raise PF4HActivationError(
            (
                ActivationBlocker(
                    "M15-ENV-002",
                    f"{RING_SLOTS_ENV}={raw!r} is not exactly '1' or '2'",
                    "unset it for the two-slot default",
                ),
            )
        )
    return int(raw)


# --------------------------------------------------------------------------
# Host-side attestation of the device entry guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptorAttestation:
    """Everything the host can prove about a built descriptor."""

    bucket_key: tuple[int, int, int, int]
    config_word: int
    descriptor_words: int
    symmetric_offsets: Mapping[str, int]
    checked: tuple[str, ...]


def _blocker(code: str, summary: str, evidence: str) -> ActivationBlocker:
    return ActivationBlocker(code, summary, evidence)


def attest_shape(bucket: M15GraphBucket) -> list[ActivationBlocker]:
    """Every shape condition the device re-checks in its entry guard.

    Mirrors k0pf6gm_device_tile_m15.hip lines 522..557 exactly.  Raising here
    is the *only* recoverable outcome: the device path sets a ``pperr`` bit
    and returns, which for a captured graph means every later replay silently
    produces nothing.
    """

    b = bucket
    out: list[ActivationBlocker] = []
    if b.topk != 8:
        out.append(
            _blocker(
                "M15-GUARD-001",
                f"TOPK={b.topk}",
                "the frozen TOPK==8 entry guard",
            )
        )
    if not 0 < b.local_experts <= K0P6_MAXE:
        out.append(
            _blocker(
                "M15-GUARD-002",
                f"E={b.local_experts} outside (0, {K0P6_MAXE}]",
                "0 < E <= K0P6_MAXE",
            )
        )
    if b.world != 8:
        out.append(
            _blocker(
                "M15-GUARD-003",
                f"world={b.world}",
                "the frozen world==8 entry guard",
            )
        )
    if b.t_loc_max <= 0 or b.t_loc_max > 0x00FFFFFF:
        out.append(
            _blocker(
                "M15-GUARD-004",
                f"T_loc_max={b.t_loc_max} outside (0, 0x00FFFFFF]",
                "a 24-bit receive-row capacity",
            )
        )
    if b.pad_max <= 0:
        out.append(
            _blocker("M15-GUARD-005", f"PADMAX={b.pad_max}", "PADMAX > 0")
        )
    if b.maxtok <= 0 or b.maxtok > b.t_loc_max // 8:
        out.append(
            _blocker(
                "M15-GUARD-006",
                f"MAXTOK={b.maxtok} exceeds T_loc_max/8={b.t_loc_max // 8}",
                "MAXTOK <= T_loc_max / 8",
            )
        )
    if not 0 <= b.tokens_per_rank <= b.maxtok:
        out.append(
            _blocker(
                "M15-GUARD-007",
                f"T={b.tokens_per_rank} outside [0, MAXTOK={b.maxtok}]",
                "the launch-constant token count",
            )
        )
    if b.chunks_per_segment > K0P6C_NCHUNK_MAX:
        out.append(
            _blocker(
                "M15-GUARD-008",
                f"(MAXTOK+{K0P6_CHUNK}-1)/{K0P6_CHUNK}="
                f"{b.chunks_per_segment} > {K0P6C_NCHUNK_MAX}",
                "the chunk-release table capacity",
            )
        )
    if b.pad_max // 32 > 16383:
        out.append(
            _blocker(
                "M15-GUARD-009",
                f"PADMAX/32={b.pad_max // 32} > 16383",
                "the 14-bit tile-event block field",
            )
        )
    if b.maxtok & (b.maxtok - 1):
        out.append(
            _blocker(
                "M15-GUARD-010",
                f"MAXTOK={b.maxtok} is not a power of two",
                "the M7 epilogue's shift/mask row decode",
            )
        )
    if b.world * b.maxtok + K0P6_M15_SLABS > b.t_loc_max:
        out.append(
            _blocker(
                "M15-GUARD-011",
                f"world*MAXTOK+{K0P6_M15_SLABS}="
                f"{b.world * b.maxtok + K0P6_M15_SLABS} > "
                f"T_loc_max={b.t_loc_max}",
                "two spare slab cells per row_ready segment",
            )
        )
    if b.pad_max % 32:
        out.append(
            _blocker(
                "M15-GUARD-012",
                f"PADMAX={b.pad_max} is not a multiple of 32",
                "a padded-sort capacity in whole 32-row blocks",
            )
        )
    minimum_pad = (
        b.world * b.tokens_per_rank * b.topk
        + 31 * b.local_experts
    )
    if b.pad_max < minimum_pad:
        out.append(
            _blocker(
                "M15-GUARD-013",
                f"PADMAX={b.pad_max} below the structural minimum "
                f"{minimum_pad}",
                "world*T*TOPK + 31*E",
            )
        )
    return out


def attest_config(word: int) -> list[ActivationBlocker]:
    """Mode/config-word half of the entry guard (lines 537..556)."""

    out: list[ActivationBlocker] = []
    config = decode_config(word)
    if not config_is_valid(config):
        out.append(
            _blocker(
                "M15-GUARD-020",
                f"packed config {word:#x} -> {config!r} fails config_is_valid",
                "a config the flags-off build accepts",
            )
        )
    if not mode_is_direct_accum(config):
        out.append(
            _blocker(
                "M15-GUARD-021",
                f"mode={config.mode} is not a direct-accumulate mode",
                "mode 12 (remote accumulate); M15 has no other transport",
            )
        )
    if config.mode != K_MODE_REMOTE_ACCUM:
        out.append(
            _blocker(
                "M15-GUARD-022",
                f"mode={config.mode}; the serving arm is pinned to mode 12",
                "K0_MPS_CFG mode=12",
            )
        )
    if encode_config(config) != word:
        out.append(
            _blocker(
                "M15-GUARD-023",
                f"config word {word:#x} does not round-trip through the codec",
                "a word produced by encode_config, never a hand-written value",
            )
        )
    if config.reserved_comm_ctas >= M15_GRID:
        out.append(
            _blocker(
                "M15-GUARD-024",
                f"C={config.reserved_comm_ctas} leaves no compute CTAs at "
                f"grid {M15_GRID}",
                "C < 256",
            )
        )
    return out


def attest_addresses(
    addresses: Mapping[str, int],
) -> list[ActivationBlocker]:
    """Alignment and nullness the device guard re-checks on the raw words.

    ``addresses`` maps the buffer names used by :func:`buffer_table` (plus
    ``heap_descriptor``) to their device addresses.
    """

    out: list[ActivationBlocker] = []
    required_alignment = {
        "heap_descriptor": 8,   # symmetric_address & 7
        "mps_state": 8,         # K0P6_D_MPS_STATE & 7
        "mps_slots": 16,        # K0P6_D_MPS_SLOTS & 15 (packet16 stores)
        "mps_queue": 4,
        "mps_arrivals": 4,
        "mps_pushed": 4,
        "mps_claim": 4,
    }
    non_null = tuple(required_alignment) + ("row_ready",)
    for name in non_null:
        address = addresses.get(name)
        if address is None:
            out.append(
                _blocker(
                    "M15-GUARD-030",
                    f"descriptor buffer {name!r} was never bound",
                    "a complete M15 buffer set",
                )
            )
            continue
        if address == 0:
            out.append(
                _blocker(
                    "M15-GUARD-031",
                    f"descriptor buffer {name!r} is null",
                    "a live device allocation",
                )
            )
    for name, alignment in required_alignment.items():
        address = addresses.get(name, 0)
        if address and address % alignment:
            out.append(
                _blocker(
                    "M15-GUARD-032",
                    f"{name} address {address:#x} is not {alignment}-byte "
                    "aligned",
                    f"a {alignment}-byte aligned allocation",
                )
            )
    return out


def attest_descriptor_words(
    words: list[int] | tuple[int, ...],
    bucket: M15GraphBucket,
    *,
    rank: int,
    config_word: int,
    spin_limit: int = M15_SPIN_LIMIT,
    part_elided: bool = False,
    expected_words: int = M15_DESCRIPTOR_WORDS,
) -> list[ActivationBlocker]:
    """Check the scalar slots of a fully built descriptor vector.

    ``part_elided`` exempts slot 21 from the non-null requirement, and *only*
    slot 21: the elision is legal solely when :func:`part_is_dead` holds, and
    the caller is required to have proved that before setting this.

    ``expected_words`` is 63 for the M15 build and 64 for the M18 replication
    arm (slot 63 = the M18R table pointer, checked non-null below like every
    other pointer slot the enum names -- slot 63 itself is outside the enum
    and validated by the device entry guard's magic/nrep check instead).
    """

    out: list[ActivationBlocker] = []
    if len(words) != expected_words:
        out.append(
            _blocker(
                "M15-DESC-001",
                f"descriptor has {len(words)} words",
                f"exactly {expected_words} int64 words for this build arm",
            )
        )
        return out
    if expected_words > M15_DESCRIPTOR_WORDS and int(words[63]) == 0:
        out.append(
            _blocker(
                "M15-DESC-005",
                "M18 descriptor slot 63 (replication table) is null",
                "a bound M18R table before any descriptor commit",
            )
        )
    if expected_words > 64 and int(words[64]) == 0:
        out.append(
            _blocker(
                "M15-DESC-006",
                "M19 descriptor slot 64 (rep_dec) is null",
                "the symmetric decision words carved with the ring slot",
            )
        )
    if expected_words > 65:
        # M20: slot 65 (DECIN) and the four pool bases (66..69) must be
        # bound; the device entry guard refuses them independently.  Slot 70
        # (prefetch duty) is deliberately nullable — cache mode passes 0 and
        # the K0P6_M20_PF_CTAS service CTAs no-op on it.
        if int(words[65]) == 0 or int(words[65]) % 8:
            out.append(
                _blocker(
                    "M15-DESC-007",
                    f"M20 descriptor slot 65 (DECIN) is {int(words[65]):#x}",
                    "an 8-byte-aligned per-layer decision bitmap buffer",
                )
            )
        for offset, name in zip(
            range(66, 70), ("W13P", "S13P", "W2P", "S2P")
        ):
            if int(words[offset]) == 0:
                out.append(
                    _blocker(
                        "M15-DESC-008",
                        f"M20 descriptor slot {offset} ({name}) is null",
                        "pool bases pointing at the cached layer's "
                        "extended-weight replica region",
                    )
                )
    expected = {
        DescriptorSlot.CUR: rank,
        DescriptorSlot.WORLD: bucket.world,
        DescriptorSlot.SPIN: spin_limit,
        DescriptorSlot.MAXB: bucket.maxb,
        DescriptorSlot.T: bucket.tokens_per_rank,
        DescriptorSlot.TOPK: bucket.topk,
        DescriptorSlot.E: bucket.local_experts,
        DescriptorSlot.TLOCMAX: bucket.t_loc_max,
        DescriptorSlot.PADMAX: bucket.pad_max,
        DescriptorSlot.DEBUG: 0,
        DescriptorSlot.MAXTOK: bucket.maxtok,
        DescriptorSlot.MPS_CFG: config_word,
    }
    for slot, value in expected.items():
        observed = int(words[int(slot)])
        if observed != value:
            out.append(
                _blocker(
                    "M15-DESC-002",
                    f"slot {slot.name}({int(slot)})={observed!r}, expected "
                    f"{value!r}",
                    "the frozen M15 scalar slot table",
                )
            )
    for slot in DescriptorSlot:
        if slot in SCALAR_SLOTS:
            continue
        if slot is DescriptorSlot.PART and part_elided:
            if int(words[int(slot)]) != 0:
                out.append(
                    _blocker(
                        "M15-DESC-004",
                        f"part was elided but slot 21 holds "
                        f"{int(words[int(slot)]):#x}",
                        "a null word, matching the kernel's own internal "
                        "nullptr on the direct-accumulate branch",
                    )
                )
            continue
        if int(words[int(slot)]) == 0:
            out.append(
                _blocker(
                    "M15-DESC-003",
                    f"pointer slot {slot.name}({int(slot)}) is null",
                    "every descriptor pointer bound before capture",
                )
            )
    return out


def refuse_if(blockers: list[ActivationBlocker]) -> None:
    if blockers:
        raise PF4HActivationError(blockers)
