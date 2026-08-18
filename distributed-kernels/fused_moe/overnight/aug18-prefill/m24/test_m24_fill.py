#!/usr/bin/env python3
"""CPU-only tests for the M24 fill-vector host plumbing and its kernel mirror.

No GPU, no node, no torch.  What this proves:

  1. ENCODE/VALIDATE PARITY -- the host encoder (``m24_fill.encode_fill_header``)
     produces payloads the Python mirror of ``k0p6_m24_publish`` accepts, and
     every corruption produces the reject reason the device's ``if/else if``
     chain would produce first.
  2. CHECKSUM DEFINITION -- host and device agree, including the rev-3 fix that
     word 0 is NOT in the sum.
  3. DESCRIPTOR SHAPE -- ``descriptor_with_fill_slot`` builds exactly 72 words,
     zero-fills the slots this build's cascade does not claim, and refuses a
     null / mis-aligned pointer and an over-long cascade.
  4. COMPOSITION VALIDATION -- ``attest_descriptor_words`` accepts plain M24,
     M24+STAGED, M24+ADAPTIVE and M24+SLOTPOOL, and refuses a descriptor with
     junk in an unclaimed slot or with ``cascade_words`` omitted.  This is the
     regression net for the rev-3 R.6/R.8b finding (the old ``m24_plain``
     switch made every PARTIAL composition unactivatable).
  5. BUFFER SPEC -- ``buffer_table(m24=True)`` yields the per-layer spec with
     ``zero=True``, which is what stops the first launch reading allocator
     residue as ``last_seen_gen``.
  6. KERNEL SOURCE INVARIANTS -- the eight rev-3 review fixes are asserted
     against the .hip text, so a later edit that reverts one fails here rather
     than on 8 GPUs.

Run:  python3 test_m24_fill.py          (also works under pytest)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
AUG18 = HERE.parents[1]
FUSED_MOE = HERE.parents[3]
KERNEL = FUSED_MOE / "k0pf6gm_device_tile_m15.hip"
SHIM = AUG18 / "m15_eplb0"

# ``m15_contracts`` is a member of the deployed ``pf4h_integration`` package on
# the node and imports two siblings (``contracts``, ``errors``) that live only
# there.  Stub exactly the two symbols it uses so the module loads on a laptop.
import importlib.util  # noqa: E402
import types  # noqa: E402
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class _FrozenContract:
    ep_size: int = 8
    topk: int = 8
    local_experts: int = 32
    hidden_size: int = 7168
    quant_block: int = 128
    global_experts: int = 256


@dataclass(frozen=True)
class _ActivationBlocker:
    code: str
    summary: str
    evidence: str


class _PF4HActivationError(RuntimeError):
    pass


def _load_shim_module(name: str):
    pkg = sys.modules.setdefault("m15_eplb0", types.ModuleType("m15_eplb0"))
    pkg.__path__ = [str(SHIM)]  # type: ignore[attr-defined]
    stub_contracts = types.ModuleType("m15_eplb0.contracts")
    stub_contracts.FROZEN_CONTRACT = _FrozenContract()  # type: ignore[attr-defined]
    stub_errors = types.ModuleType("m15_eplb0.errors")
    stub_errors.ActivationBlocker = _ActivationBlocker  # type: ignore[attr-defined]
    stub_errors.PF4HActivationError = _PF4HActivationError  # type: ignore[attr-defined]
    sys.modules.setdefault("m15_eplb0.contracts", stub_contracts)
    sys.modules.setdefault("m15_eplb0.errors", stub_errors)
    spec = importlib.util.spec_from_file_location(
        f"m15_eplb0.{name}", SHIM / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"m15_eplb0.{name}"] = module
    spec.loader.exec_module(module)
    return module


C = _load_shim_module("m15_contracts")
F = _load_shim_module("m24_fill")

WORLD = 8
MAXTOK = 4096
T_CAP = 4096
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def _buffer(header: list[int], last_gen: int = 0) -> list[int]:
    """Header + n_orig + the 4-word device-owned tail."""
    return list(header) + [last_gen, 0, 0, 0]


# ---------------------------------------------------------------- 1 + 2
def test_encode_validate_parity() -> None:
    print("1/2. encode <-> validate parity, and the checksum definition")
    n_orig = [1539, 1600, 1480, 1720, 0, 4096, 900, 2048]

    header = F.encode_fill_header(1, n_orig, world=WORLD)
    check("header length is 8 + world", len(header) == 8 + WORLD)
    check("word 0 is the magic", header[0] == C.M24_FILL_MAGIC)
    check("gen_echo duplicates gen", header[4] == header[1] == 1)
    check("reserved words 5..7 are zero", header[5:8] == [0, 0, 0])

    # The rev-3 checksum: seeded with the magic, word 0 NOT xored in.
    expect = C.M24_FILL_MAGIC ^ 1 ^ 1
    for value in n_orig:
        expect ^= value
    check("csum matches the seeded definition", header[3] == expect & 0xFFFFFFFF,
          f"got {header[3]:#x} want {expect & 0xFFFFFFFF:#x}")
    check(
        "csum is NOT the self-cancelling variant",
        header[3] != (expect ^ C.M24_FILL_MAGIC) & 0xFFFFFFFF,
    )

    reason, t_eff = F.validate_payload_mirror(
        _buffer(header), cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
    )
    check("a fresh payload is accepted", reason == F.RejectReason.ACCEPT)
    check("T_eff = round_up(1539, 256) = 1792", t_eff == 1792, f"got {t_eff}")

    # Every reject reason, in the device's evaluation order.
    cases = [
        ("magic", lambda w: w.__setitem__(0, 0xDEADBEEF), F.RejectReason.MAGIC),
        ("tear", lambda w: w.__setitem__(4, 99), F.RejectReason.TEAR),
        ("csum", lambda w: w.__setitem__(3, w[3] ^ 1), F.RejectReason.CSUM),
        ("flags", lambda w: w.__setitem__(2, 0), F.RejectReason.FLAGS),
        ("range", lambda w: w.__setitem__(8, MAXTOK + 1), F.RejectReason.RANGE),
    ]
    for name, corrupt, want in cases:
        words = _buffer(F.encode_fill_header(7, n_orig, world=WORLD))
        if name in ("flags", "csum", "range"):
            # flags/range change the payload, so re-seal the checksum where the
            # corruption is meant to be caught by a LATER test than csum.
            if name == "flags":
                words[2] = 0
                words[3] = C.m24_fill_checksum(7, 0, words[8 : 8 + WORLD])
            elif name == "range":
                bad = list(n_orig)
                bad[0] = MAXTOK + 1
                words = _buffer(F.encode_fill_header(7, bad, world=WORLD))
            else:
                corrupt(words)
        else:
            corrupt(words)
        reason, t_eff = F.validate_payload_mirror(
            words, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
        )
        check(f"reject reason {name}", reason == want, f"got {reason}")
        check(f"{name} degrades to the padded capacity", t_eff == T_CAP)

    # Stale: gen <= last_seen_gen.  This is the EXPECTED path the design
    # enumerates, and it must degrade to T_cap, never to a short step.
    words = _buffer(F.encode_fill_header(5, n_orig, world=WORLD), last_gen=5)
    reason, t_eff = F.validate_payload_mirror(
        words, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
    )
    check("replayed gen is stale", reason == F.RejectReason.STALE)
    check("stale degrades to the padded capacity", t_eff == T_CAP)

    words = _buffer(F.encode_fill_header(6, n_orig, world=WORLD), last_gen=5)
    reason, _ = F.validate_payload_mirror(
        words, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
    )
    check("gen = last + 1 is accepted", reason == F.RejectReason.ACCEPT)

    # Per-rank selection and the saturating clamp.
    for cur, want in ((5, 4096), (4, 0), (7, 2048)):
        _, t_eff = F.validate_payload_mirror(
            _buffer(F.encode_fill_header(3, n_orig, world=WORLD)),
            cur=cur, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP,
        )
        check(f"cur={cur} selects its own n_orig", t_eff == want, f"got {t_eff}")


# ---------------------------------------------------------------- 3
def test_descriptor_shape() -> None:
    print("3. descriptor_with_fill_slot")
    for cascade_len in (63, 64, 65, 71):
        cascade = list(range(1, cascade_len + 1))
        words = F.descriptor_with_fill_slot(cascade, 0x1000)
        check(f"cascade {cascade_len} -> 72 words", len(words) == 72)
        check(
            f"cascade {cascade_len} keeps its own slots",
            words[:cascade_len] == cascade,
        )
        check(
            f"cascade {cascade_len} zero-fills the unclaimed slots",
            all(w == 0 for w in words[cascade_len:71]),
        )
        check(f"cascade {cascade_len} binds slot 71", words[71] == 0x1000)

    for bad_ptr in (0, 0x1008, 0x1001):
        try:
            F.descriptor_with_fill_slot(list(range(63)), bad_ptr)
            check(f"pointer {bad_ptr:#x} refused", False, "no exception")
        except ValueError:
            check(f"pointer {bad_ptr:#x} refused", True)
    try:
        F.descriptor_with_fill_slot(list(range(72)), 0x1000)
        check("over-long cascade refused", False, "no exception")
    except ValueError:
        check("over-long cascade refused", True)


# ---------------------------------------------------------------- 4
def test_composition_validation() -> None:
    print("4. attest_descriptor_words across the compositions")
    bucket = C.M15_PREFILL_B4096_BUCKET

    def descriptor(cascade_len: int, *, fill_ptr: int = 0x2000,
                   junk_slot: int | None = None) -> list[int]:
        # Non-null plausible words for every slot the cascade claims; the
        # scalar-slot checks below are exercised by the module's own suite, so
        # here we only care about the 63..71 window and the length.
        words = [0x40 * (i + 1) for i in range(cascade_len)]
        words = F.descriptor_with_fill_slot(words, fill_ptr)
        if junk_slot is not None:
            words[junk_slot] = 0xBADF00D0
        return words

    def codes(cascade_len: int, **kwargs) -> set[str]:
        words = descriptor(cascade_len, **kwargs)
        blockers = C.attest_descriptor_words(
            words,
            bucket,
            rank=0,
            config_word=0,
            expected_words=C.M15_DESCRIPTOR_WORDS_M24,
            cascade_words=cascade_len,
        )
        return {b.code for b in blockers}

    for cascade_len, label in (
        (63, "plain M24"),
        (64, "M24 + STAGED/REPLICATE"),
        (65, "M24 + ADAPTIVE"),
        (71, "M24 + SLOTPOOL"),
    ):
        found = codes(cascade_len)
        spurious = found & {
            "M15-DESC-005", "M15-DESC-006", "M15-DESC-007", "M15-DESC-008",
            "M15-DESC-013", "M15-DESC-014", "M15-DESC-015",
        }
        check(f"{label} raises no M24/M18/M19/M20 slot blocker",
              not spurious, f"got {sorted(spurious)}")

    check(
        "junk in an unclaimed slot is refused",
        "M15-DESC-015" in codes(63, junk_slot=64),
    )
    words = descriptor(63)
    words[71] = 0
    blockers = C.attest_descriptor_words(
        words, bucket, rank=0, config_word=0,
        expected_words=C.M15_DESCRIPTOR_WORDS_M24, cascade_words=63,
    )
    check("a null fill pointer is refused",
          "M15-DESC-013" in {b.code for b in blockers})
    words[71] = 0x2008
    blockers = C.attest_descriptor_words(
        words, bucket, rank=0, config_word=0,
        expected_words=C.M15_DESCRIPTOR_WORDS_M24, cascade_words=63,
    )
    check("an 8-byte-aligned fill pointer is refused",
          "M15-DESC-013" in {b.code for b in blockers})

    blockers = C.attest_descriptor_words(
        descriptor(63), bucket, rank=0, config_word=0,
        expected_words=C.M15_DESCRIPTOR_WORDS_M24,
    )
    check("omitting cascade_words on an M24 build is refused",
          "M15-DESC-014" in {b.code for b in blockers})

    # A default-0 (non-M24) arm must walk exactly the slots it walked before.
    non_m24 = [0x40 * (i + 1) for i in range(63)]
    blockers = C.attest_descriptor_words(
        non_m24, bucket, rank=0, config_word=0
    )
    check("a default-0 arm raises no M24 blocker",
          not ({"M15-DESC-013", "M15-DESC-014", "M15-DESC-015"}
               & {b.code for b in blockers}))


# ---------------------------------------------------------------- 5
def test_buffer_spec() -> None:
    print("5. buffer_table(m24=True)")
    bucket = C.M15_PREFILL_B4096_BUCKET
    try:
        table = C.buffer_table(bucket, m24=True)
    except TypeError as exc:
        check("buffer_table accepts m24=", False, str(exc))
        return
    specs = [s for s in table if s.name == "m24_fill"]
    check("the m24_fill spec is present", len(specs) == 1)
    if not specs:
        return
    spec = specs[0]
    check("slot 71", int(spec.slot) == 71)
    check("int32", spec.itemsize == 4 and spec.dtype == "int32")
    check("not symmetric (rank-local)", spec.symmetric is False)
    check("ZEROED before the first epoch", spec.zero is True)
    check("8 + world + 4 words",
          spec.elements == 8 + WORLD + 4, f"got {spec.elements}")
    check("32 + 4*world + 16 bytes",
          spec.nbytes == 32 + 4 * WORLD + 16, f"got {spec.nbytes}")
    table0 = C.buffer_table(bucket)
    check("a default-0 arm allocates no fill buffer",
          not any(s.name == "m24_fill" for s in table0))


# ---------------------------------------------------------------- 6
def test_kernel_invariants() -> None:
    print("6. kernel source invariants (the rev-3 review fixes)")
    src = KERNEL.read_text()

    publish = src[src.index("k0p6_m24_publish(const long long* desc"):]
    publish = publish[: publish.index("\n}\n")]

    check(
        "R.1 the publish releases before the caller's barrier arrive",
        "release_signal_batch_agent()" in publish,
    )
    check(
        "R.2 STRICT never raises the grid-return bit 16777216",
        "atomicOr(pperr, 16777216)" not in publish,
    )
    check(
        "R.2 STRICT raises its own telemetry bit",
        "atomicOr(pperr, K0P6_M24_ERR_STRICT)" in publish,
    )
    check(
        "R.3 the NORIG block no longer kills the validation path",
        not re.search(r"^\s*reason = 0u;", publish, re.M),
    )
    check(
        "R.3 the substitution is gated on an accepted payload",
        re.search(r"NORIG_TABLE\) \|\| K0P6_M24_NORIG_CONST\n(.|\n)*?"
                  r"if \(reason == 0u\) \{", publish) is not None,
    )
    check("R.7a NORIG_TABLE is a select chain, not tbl[cur]",
          "k0p6_m24_tbl[cur]" not in src)
    asm_lines = [ln for ln in publish.splitlines()
                 if "asm volatile" in ln and not ln.lstrip().startswith("//")]
    check('R.7b the opacity barrier uses "+v", not "+s"',
          len(asm_lines) == 1 and '"+v"(n_sub)' in asm_lines[0]
          and '"+s"' not in asm_lines[0],
          f"asm lines: {asm_lines}")

    teff = src[src.index("int k0p6_m24_teff(const long long* desc"):]
    teff = teff[: teff.index("\n}\n")]
    check(
        "R.4 the consumer clamps T_eff to the padded capacity",
        "(unsigned int)t > (unsigned int)T_cap" in teff,
    )

    check("R.5 the reject bit is 1<<29, not K0P6_MPS_ERR_SERVICE",
          "#define K0P6_M24_ERR_REJECT 536870912" in src)
    check("R.5 the strict bit is 1<<30",
          "#define K0P6_M24_ERR_STRICT 1073741824" in src)
    for stale in ("67108864", "268435456", "134217728", "33554432"):
        check(f"R.5 no M24 bit aliases {stale}",
              f"K0P6_M24_ERR_REJECT {stale}" not in src
              and f"K0P6_M24_ERR_STRICT {stale}" not in src)

    check("R.8 the checksum seed no longer cancels word 0",
          "unsigned int acc = K0P6_M24_MAGIC ^ gen ^ flags;" in src)
    check("R.9a the slot check compares against the cascade length",
          "#if K0P6_M15_D_LEN > K0P6_D_M24_FILL" in src)
    directives = [ln for ln in src.splitlines() if ln.startswith("#if ")]
    check("R.9a the tautological literal check is gone",
          not any("K0P6_D_M24_FILL >= 72" in ln for ln in directives))
    check("R.9b SRC_REV is bumped on the M24 branch",
          re.search(r"#if K0P6_M24_FILL\n#undef K0P6_M15_SRC_REV\n"
                    r"#define K0P6_M15_SRC_REV 3", src) is not None)
    check("R.10 NULLWORK has a FILL cross-check",
          "#if K0P6_M24_NULLWORK && !K0P6_M24_FILL" in src)

    # Every M24 macro has an #error tying it to FILL.
    for macro in ("ZERO_PAD", "ALLOW_DIRTY_PAD", "FILL_C", "M8_ADAPT",
                  "NORIG_CONST", "STRICT", "NULLWORK"):
        check(f"cross-check for K0P6_M24_{macro}",
              f"#if K0P6_M24_{macro} && !K0P6_M24_FILL" in src)


# ---------------------------------------------------------------- 7
class _FakeView:
    def __init__(self, parent: "_FakeTensor", start: int, stop: int) -> None:
        self.parent, self.start, self.stop = parent, start, stop

    def copy_(self, other, non_blocking: bool = False):  # noqa: ANN001
        values = other.tolist() if hasattr(other, "tolist") else list(other)
        assert len(values) == self.stop - self.start
        self.parent.data[self.start : self.stop] = values
        return self

    def tolist(self) -> list[int]:
        return list(self.parent.data[self.start : self.stop])


class _FakeTensor:
    """The 1-D int32 slice of torch this module actually uses."""

    _next_ptr = 0x7F0000000000

    def __init__(self, n: int) -> None:
        self.data = [0] * n
        _FakeTensor._next_ptr += 256          # torch's allocator base grain
        self._ptr = _FakeTensor._next_ptr

    def data_ptr(self) -> int:
        return self._ptr

    def __getitem__(self, key):  # noqa: ANN001
        if isinstance(key, slice):
            start = key.start or 0
            stop = len(self.data) if key.stop is None else key.stop
            return _FakeView(self, start, stop)
        return self.data[key]

    def __setitem__(self, key, value):  # noqa: ANN001
        self.data[key] = value

    def copy_(self, other, non_blocking: bool = False):  # noqa: ANN001
        self.data[:] = other.tolist() if hasattr(other, "tolist") else list(other)
        return self

    def tolist(self) -> list[int]:
        return list(self.data)


class _FakeTorch:
    int32 = "int32"

    @staticmethod
    def zeros(n, dtype=None, device=None, pin_memory=False):  # noqa: ANN001
        return _FakeTensor(n)

    @staticmethod
    def empty(n, dtype=None, device=None, pin_memory=False):  # noqa: ANN001
        return _FakeTensor(n)

    @staticmethod
    def tensor(values, dtype=None):  # noqa: ANN001
        t = _FakeTensor(len(values))
        t.data[:] = list(values)
        return t


def test_fill_vector_writes() -> None:
    print("7. M24FillVector: allocation, the per-launch write, the fast path")
    fill = F.M24FillVector(
        world=WORLD, num_layers=2, maxtok=MAXTOK, torch_module=_FakeTorch
    )
    check("buffer is 32 + 4*world + 16 bytes", fill.nbytes() == 32 + 4 * WORLD + 16)
    check("pointers are 16-byte aligned",
          all(p % 16 == 0 for p in fill.pointers()))
    check("the device-owned tail starts zeroed",
          fill.read_tail(0) == {"last_seen_gen": 0, "agreed_T_eff": 0,
                                "agreed_flags": 0})

    n_orig = [1539, 1600, 1480, 1720, 0, 4096, 900, 2048]
    last_gen = 0
    # 4 launches: the first takes the full-encode path, the rest the 3-scalar
    # fast path.  Every one of them must be ACCEPTED by the device mirror
    # against the last_seen_gen the previous launch would have committed.
    for launch in range(4):
        gen = fill.write(0, n_orig)
        raw = [w & 0xFFFFFFFF for w in fill._layers[0].tensor.tolist()]
        raw[8 + WORLD] = last_gen           # what the device committed
        reason, t_eff = F.validate_payload_mirror(
            raw, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
        )
        check(f"launch {launch} payload accepted",
              reason == F.RejectReason.ACCEPT, f"reason {reason}")
        check(f"launch {launch} T_eff = 1792", t_eff == 1792, f"got {t_eff}")
        check(f"launch {launch} gen advanced", gen == launch + 1)
        last_gen = gen

    # Replaying a launch without bumping gen must be rejected as stale -- the
    # consume-once contract that stops a graph replay running on a payload the
    # pre-op never refreshed.
    fill.write(0, n_orig, gen=last_gen)
    raw = [w & 0xFFFFFFFF for w in fill._layers[0].tensor.tolist()]
    raw[8 + WORLD] = last_gen
    reason, t_eff = F.validate_payload_mirror(
        raw, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
    )
    check("a replayed gen is stale", reason == F.RejectReason.STALE)
    check("stale still degrades to the padded capacity", t_eff == T_CAP)

    # Changing the payload must take the full-encode path and stay valid.
    other = [4096] * WORLD
    fill.write(1, other)
    raw = [w & 0xFFFFFFFF for w in fill._layers[1].tensor.tolist()]
    reason, t_eff = F.validate_payload_mirror(
        raw, cur=3, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
    )
    check("a changed payload re-encodes correctly",
          reason == F.RejectReason.ACCEPT)
    check("full fill yields T_eff = T_cap", t_eff == T_CAP)
    check("layers keep independent gens",
          fill.current_gen(0) == 4 and fill.current_gen(1) == 1,
          f"got {fill.current_gen(0)}, {fill.current_gen(1)}")

    # Interleaving two layers on the SAME payload exercises the fast path with
    # a per-layer gen: the shared staging buffer must be re-stamped each time.
    for _ in range(3):
        fill.write(0, other)
        fill.write(1, other)
    for layer, prev in ((0, fill.current_gen(0) - 1), (1, fill.current_gen(1) - 1)):
        raw = [w & 0xFFFFFFFF for w in fill._layers[layer].tensor.tolist()]
        raw[8 + WORLD] = prev
        reason, _ = F.validate_payload_mirror(
            raw, cur=0, world=WORLD, maxtok=MAXTOK, t_cap=T_CAP
        )
        check(f"interleaved layer {layer} stays accepted",
              reason == F.RejectReason.ACCEPT, f"reason {reason}")


def main() -> int:
    test_encode_validate_parity()
    test_descriptor_shape()
    test_composition_validation()
    test_buffer_spec()
    test_kernel_invariants()
    test_fill_vector_writes()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("all M24 host-plumbing tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
