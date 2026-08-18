"""M24 fill-vector host plumbing: allocate, bind, and write the payload.

This is the host half of ``FILL_AWARE_DESIGN.md`` section B.3 (work-list G12 /
G13b).  It was entirely missing from the first M24 implementation, which is why
both adversarial reviews returned a blocking finding: the kernel's
``K0P6_M24_FILL=1`` path reads descriptor slot 71 and dereferences it, and
nothing in either repository allocated a buffer, bound the slot, or wrote a
payload.  A 63-word descriptor plus an unconditional ``desc[71]`` read is an
out-of-bounds device read 72 bytes past the tensor, and -- because
``k0p6_m24_publish`` also STORES through that pointer -- a wild device write on
all 8 GPUs whenever the residue happened to look like a 16-byte-aligned
non-null address.

Three consumers, one definition:

* the MoK harness (``m24_mok_patch.py`` wires this module into
  ``e004pf_k0pf_ab.py``),
* the serving shim (``m15_runtime.py`` on the node, work-list G13b -- which
  writes the same record from a device kernel instead of a host copy so the
  op is capturable; the ENCODING below is what that kernel must produce),
* the offline tests (``m24/test_m24_fill.py``), which re-implement the DEVICE
  validator in Python and prove the two agree.

Buffer layout (int32 words, one buffer PER LAYER, 16-byte aligned)::

    [0]            magic     0x4D323446 "M24F"
    [1]            gen       monotonic, >= 1, one per mega-eligible model step
    [2]            flags     bit0 = fill-aware requested this step
    [3]            csum      MAGIC ^ gen ^ flags ^ n_orig[0..world)
    [4]            gen_echo  duplicate of gen (tear detector)
    [5..7]         reserved (zero)
    [8 + p]        n_orig[p] for p in range(world)
    ---- DEVICE-OWNED TAIL: the host writes it exactly ONCE, at allocation,
         by zeroing.  Never again. ----
    [8+world+0]    last_seen_gen   per-layer consume-once token
    [8+world+1]    agreed_T_eff    the grid-uniform broadcast
    [8+world+2]    agreed_flags
    [8+world+3]    reserved

The zeroing is load-bearing, not hygiene.  ``last_seen_gen`` is READ by the
very first launch and written only on an accepted step, so allocator residue
>= the first ``gen`` (0xDEADBEEF out of a recycled caching-allocator block is
the common case) makes every step reject with ``RJ_STALE``, and the arm
silently runs the pre-M24 kernel while reporting itself as M24 -- the project's
named past sin, with no receipt able to detect it.

This module deliberately imports ``torch`` lazily so the encoding, the
checksum and the validator mirror can be tested on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: ``K0P6_M24_MAGIC`` -- ASCII ``"M24F"``.
M24_FILL_MAGIC = 0x4D323446
#: Header words before the ``n_orig`` vector.
M24_FILL_HEADER_WORDS = 8
#: The DEVICE-OWNED tail the host writes exactly once, by zeroing.
M24_FILL_TAIL_WORDS = 4

# DELIBERATELY SELF-CONTAINED.  ``m15_contracts`` is a member of the deployed
# ``pf4h_integration`` package and imports two siblings that exist only on the
# node, so importing it here would make this module unusable from the MoK
# harness and from a laptop test.  The definitions are duplicated instead and
# CROSS-CHECKED below whenever m15_contracts does happen to be importable, so
# a drift between the two is a loud AssertionError rather than a silent
# checksum mismatch that the device reports as RJ_CSUM.


def m24_fill_buffer_words(world: int) -> int:
    """int32 words in one layer's M24 fill buffer (header + n_orig + tail)."""
    return M24_FILL_HEADER_WORDS + int(world) + M24_FILL_TAIL_WORDS


def m24_fill_buffer_bytes(world: int) -> int:
    """Bytes in one layer's M24 fill buffer: ``32 + 4*world + 16``."""
    return 4 * m24_fill_buffer_words(world)


def m24_fill_checksum(
    gen: int, flags: int, n_orig: "Sequence[int]"
) -> int:
    """``csum``: seed with the magic, xor gen, flags and every ``n_orig``.

    Word 0 is NOT in the sum: it IS the magic on any payload that clears the
    magic test, so xoring it in would cancel the seed exactly (rev 3 review
    fix R.8).  Mirrored by ``k0p6_m24_publish``.
    """
    acc = M24_FILL_MAGIC
    acc ^= int(gen) & 0xFFFFFFFF
    acc ^= int(flags) & 0xFFFFFFFF
    for value in n_orig:
        acc ^= int(value) & 0xFFFFFFFF
    return acc & 0xFFFFFFFF


def _cross_check_contracts() -> None:
    """Assert this module and ``m15_contracts`` cannot drift apart."""
    try:
        from .m15_contracts import (  # type: ignore[import-not-found]
            M24_FILL_HEADER_WORDS as _H,
            M24_FILL_MAGIC as _M,
            M24_FILL_TAIL_WORDS as _T,
            m24_fill_checksum as _csum,
        )
    except Exception:  # noqa: BLE001 - any import failure means "not here"
        return
    assert (_M, _H, _T) == (
        M24_FILL_MAGIC,
        M24_FILL_HEADER_WORDS,
        M24_FILL_TAIL_WORDS,
    ), "m24_fill and m15_contracts disagree on the fill-buffer layout"
    probe = [1539, 0, 4096, 7, 7, 7, 7, 7]
    assert _csum(3, 1, probe) == m24_fill_checksum(3, 1, probe), (
        "m24_fill and m15_contracts disagree on the fill-vector checksum"
    )


_cross_check_contracts()

__all__ = [
    "M24_FILL_ALIGNMENT",
    "M24_DESCRIPTOR_WORDS",
    "M24_FILL_SLOT",
    "RejectReason",
    "encode_fill_header",
    "descriptor_with_fill_slot",
    "validate_payload_mirror",
    "M24FillVector",
]

#: The device entry guard refuses anything else (``desc[71] & 15``).
M24_FILL_ALIGNMENT = 16
#: ``K0P6_D_M24_FILL``.
M24_FILL_SLOT = 71
#: ``K0P6_M15_D_LEN`` after M24's raise-not-redefine block.
M24_DESCRIPTOR_WORDS = 72

_U32 = 0xFFFFFFFF


class RejectReason:
    """Mirror of the ``K0P6_M24_RJ_*`` codes, in the device's test order."""

    ACCEPT = 0
    MAGIC = 1
    TEAR = 2
    CSUM = 3
    FLAGS = 4
    STALE = 5
    RANGE = 6


def encode_fill_header(
    gen: int,
    n_orig: Sequence[int],
    *,
    world: int | None = None,
    flags: int = 1,
) -> list[int]:
    """The ``8 + world`` int32 words the pre-op writes.  Never the tail.

    ``gen`` must be >= 1 and strictly increasing per layer buffer: the device
    rejects ``gen <= last_seen_gen`` as stale.  ``flags`` bit 0 must be set or
    the device rejects with ``RJ_FLAGS`` -- clearing it is the deliberate
    "run this step at the padded capacity" escape.
    """
    values = [int(v) for v in n_orig]
    if world is None:
        world = len(values)
    if len(values) != world:
        raise ValueError(f"n_orig has {len(values)} entries, world is {world}")
    if int(gen) <= 0 or int(gen) > _U32:
        raise ValueError(f"gen must be in [1, 2**32-1]; got {gen}")
    if any(v < 0 or v > _U32 for v in values):
        raise ValueError(f"n_orig out of u32 range: {values}")
    gen = int(gen) & _U32
    flags = int(flags) & _U32
    words = [0] * (M24_FILL_HEADER_WORDS + world)
    words[0] = M24_FILL_MAGIC
    words[1] = gen
    words[2] = flags
    words[3] = m24_fill_checksum(gen, flags, values)
    words[4] = gen                      # gen_echo
    # words[5..7] stay zero (reserved).
    for p, value in enumerate(values):
        words[M24_FILL_HEADER_WORDS + p] = value & _U32
    return words


def descriptor_with_fill_slot(
    cascade_words: Sequence[int], fill_ptr: int
) -> list[int]:
    """Extend a cascade-length descriptor to the 72 words an M24 build reads.

    ``cascade_words`` is the descriptor the composition cascade already built
    (63 plain, 64 with STAGED/REPLICATE, 65 with +ADAPTIVE, 71 with
    +SLOTPOOL).  Slots between its end and 71 are UNCLAIMED by this build and
    are zero-filled: the kernel never reads them, and
    ``m15_contracts.attest_descriptor_words`` refuses a non-zero word there.
    """
    words = [int(w) for w in cascade_words]
    if len(words) > M24_FILL_SLOT:
        raise ValueError(
            f"cascade produced {len(words)} words, so another arm already "
            f"claims slot {M24_FILL_SLOT}; the kernel's own #error refuses "
            "this build"
        )
    if int(fill_ptr) == 0 or int(fill_ptr) % M24_FILL_ALIGNMENT:
        raise ValueError(
            f"fill buffer pointer {fill_ptr:#x} must be non-null and "
            f"{M24_FILL_ALIGNMENT}-byte aligned (the device entry guard "
            "fails closed on anything else)"
        )
    words.extend([0] * (M24_FILL_SLOT - len(words)))
    words.append(int(fill_ptr))
    assert len(words) == M24_DESCRIPTOR_WORDS
    return words


def validate_payload_mirror(
    buffer_words: Sequence[int],
    *,
    cur: int,
    world: int,
    maxtok: int,
    t_cap: int,
    tgrain: int = 256,
) -> tuple[int, int]:
    """Python mirror of ``k0p6_m24_publish``.  Returns ``(reason, T_eff)``.

    Kept in the same order as the device's ``if/else if`` chain so a test can
    assert both the accept/reject verdict AND which reason fires first.  The
    saturating round-up mirrors ``k0p6_m24_round_up_sat``.
    """
    words = [int(w) & _U32 for w in buffer_words]
    need = M24_FILL_HEADER_WORDS + world + M24_FILL_TAIL_WORDS
    if len(words) != need:
        raise ValueError(f"buffer has {len(words)} words, expected {need}")
    magic, gen, flags, csum, gen_echo = words[0:5]
    n_orig = words[M24_FILL_HEADER_WORDS : M24_FILL_HEADER_WORDS + world]
    last_gen = words[M24_FILL_HEADER_WORDS + world + 0]

    acc = M24_FILL_MAGIC ^ gen ^ flags
    range_ok = True
    n_self = 0
    for p, value in enumerate(n_orig):
        acc ^= value
        if value > maxtok:
            range_ok = False
        if p == cur:
            n_self = value
    acc &= _U32

    if magic != M24_FILL_MAGIC:
        reason = RejectReason.MAGIC
    elif gen_echo != gen:
        reason = RejectReason.TEAR
    elif acc != csum:
        reason = RejectReason.CSUM
    elif (flags & 1) == 0:
        reason = RejectReason.FLAGS
    elif gen <= last_gen:
        reason = RejectReason.STALE
    elif not range_ok:
        reason = RejectReason.RANGE
    else:
        reason = RejectReason.ACCEPT

    if reason != RejectReason.ACCEPT:
        return reason, t_cap
    return reason, _round_up_sat(n_self, tgrain, t_cap)


def _as_i32(word: int) -> int:
    """u32 -> the signed value torch.int32 stores.  The magic and most
    checksums exceed 2**31, so this is not optional."""
    word &= _U32
    return word - (1 << 32) if word >= (1 << 31) else word


def _round_up_sat(n: int, grain: int, cap: int) -> int:
    if cap <= 0 or n <= 0:
        return 0
    if n >= cap:
        return cap
    r = ((n + grain - 1) // grain) * grain
    return cap if r >= cap else r


@dataclass
class _LayerBuffer:
    tensor: object
    pointer: int
    gen: int


class M24FillVector:
    """One zeroed fill buffer per layer, plus the per-step write.

    Usage (MoK harness and shim alike)::

        fill = M24FillVector(world=WORLD, num_layers=1, maxtok=MAXTOK)
        desc = descriptor_with_fill_slot(cascade_desc, fill.pointer(0))
        ...
        fill.write(0, n_orig)           # ON THE MEGA'S STREAM, before launch
        launch_mega(desc)

    ``write`` is stream-ordered against the launch that follows it, which is
    what makes the tear detector unnecessary on the primary path -- it exists
    for a future writer that is not stream-ordered with the mega.
    """

    def __init__(
        self,
        *,
        world: int,
        num_layers: int = 1,
        maxtok: int = 4096,
        device: str = "cuda",
        torch_module: object | None = None,
    ) -> None:
        if torch_module is None:  # pragma: no cover - node-only path
            import torch as torch_module  # type: ignore[no-redef]
        self._torch = torch_module
        self.world = int(world)
        self.maxtok = int(maxtok)
        self.num_layers = int(num_layers)
        self._words = m24_fill_buffer_words(self.world)
        self._header_words = M24_FILL_HEADER_WORDS + self.world
        self._layers: list[_LayerBuffer] = []
        for _ in range(self.num_layers):
            # zeros(), never empty(): the device-owned tail's last_seen_gen is
            # read on the FIRST launch.  See the module docstring.
            tensor = self._torch.zeros(
                self._words, dtype=self._torch.int32, device=device
            )
            pointer = int(tensor.data_ptr())
            if pointer % M24_FILL_ALIGNMENT:
                raise RuntimeError(
                    f"M24 fill buffer at {pointer:#x} is not "
                    f"{M24_FILL_ALIGNMENT}-byte aligned; the device entry "
                    "guard fails closed on it"
                )
            self._layers.append(_LayerBuffer(tensor, pointer, 0))
        # Pinned host staging for the header write.  One small H2D copy per
        # launch beats a kernel launch, and it is the same order of magnitude
        # as the descriptor writes the shim already does out of capture.
        self._staging = self._torch.zeros(
            self._header_words, dtype=self._torch.int32, pin_memory=True
        )
        # The payload is constant across launches in every MoK arm and across
        # most serving steps, so the full encode runs only when it changes and
        # a repeat write touches exactly three scalars (gen, csum, gen_echo).
        # This matters: the pre-op sits INSIDE the timed region, and a
        # per-launch Python re-encode plus tensor construction would be tens of
        # microseconds of host work able to starve the device in a tight loop.
        self._cached_payload: tuple[tuple[int, ...], int] | None = None

    def nbytes(self) -> int:
        return m24_fill_buffer_bytes(self.world)

    def pointer(self, layer: int = 0) -> int:
        return self._layers[layer].pointer

    def pointers(self) -> list[int]:
        return [entry.pointer for entry in self._layers]

    def current_gen(self, layer: int = 0) -> int:
        return self._layers[layer].gen

    def write(
        self,
        layer: int,
        n_orig: Sequence[int],
        *,
        gen: int | None = None,
        flags: int = 1,
        non_blocking: bool = True,
    ) -> int:
        """Write one step's payload into ``layer``'s buffer.  Returns ``gen``.

        The default ``gen`` is this layer's previous value plus one, which is
        exactly the device's consume-once contract.  Passing ``gen`` is for
        tests that need to reproduce a stale or replayed payload.
        """
        entry = self._layers[layer]
        if gen is None:
            gen = entry.gen + 1
        key = (tuple(int(v) for v in n_orig), int(flags))
        if self._cached_payload != key:
            words = encode_fill_header(
                gen, n_orig, world=self.world, flags=flags
            )
            self._staging.copy_(
                self._torch.tensor(
                    [_as_i32(w) for w in words], dtype=self._torch.int32
                )
            )
            self._cached_payload = key
        else:
            csum = m24_fill_checksum(gen, flags, key[0])
            self._staging[1] = _as_i32(gen & _U32)
            self._staging[3] = _as_i32(csum)
            self._staging[4] = _as_i32(gen & _U32)
        entry.tensor[: self._header_words].copy_(
            self._staging, non_blocking=non_blocking
        )
        entry.gen = gen
        return gen

    def write_all(
        self, n_orig: Sequence[int], *, flags: int = 1
    ) -> list[int]:
        return [
            self.write(layer, n_orig, flags=flags)
            for layer in range(self.num_layers)
        ]

    def read_tail(self, layer: int = 0) -> dict[str, int]:
        """Host-side read of the device-owned tail, for receipts (G15).

        Synchronising; never call it on the timed path.
        """
        entry = self._layers[layer]
        tail = entry.tensor[self._header_words :].tolist()
        return {
            "last_seen_gen": int(tail[0]) & _U32,
            "agreed_T_eff": int(tail[1]) & _U32,
            "agreed_flags": int(tail[2]) & _U32,
        }
