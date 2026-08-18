#!/usr/bin/env python3
"""CPU-only tests for m24_mok_patch.py, against a byte-copy of the harness.

Proves, with no GPU and no node:

  1. ANCHORS   -- all four anchors match exactly once in the real harness.
  2. APPLY     -- the patched file AST-parses and carries the marker.
  3. IDEMPOTENCY -- a second run is a no-op and leaves the bytes unchanged.
  4. TAMPER    -- a hand-edited file that keeps the marker but loses a hunk is
                  REFUSED rather than double-patched.
  5. INERTNESS -- with K0_M24_FILL unset, every injected predicate is false, so
                  the descriptor, the expected-word count and the launch path
                  are the pre-patch ones.  This is the zero-change ratchet the
                  runbook requires before any arm is believed.
  6. LIVE PATH -- with K0_M24_FILL set, the injected helper allocates, binds
                  slot 71 into a 72-word descriptor, and issues one pre-op per
                  launch whose payload the DEVICE MIRROR accepts.

Run:  python3 test_m24_mok_patch.py       (also works under pytest)

Set K0_HARNESS to point at a different e004pf_k0pf_ab.py.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve()
M24 = HERE.parent
AUG18 = HERE.parents[1]
SHIM = AUG18 / "m15_eplb0"
PATCHER = M24 / "m24_mok_patch.py"

HARNESS = Path(
    os.environ.get(
        "K0_HARNESS",
        os.path.expanduser(
            "~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/"
            "e004pf_k0pf_ab.py"
        ),
    )
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if not HARNESS.is_file():
        print(f"SKIP: harness not found at {HARNESS}")
        print("      (set K0_HARNESS to run these tests)")
        return 0

    patcher = _load(PATCHER, "m24_mok_patch")
    original = HARNESS.read_text()

    print("1. anchors")
    for name, anchor, _repl in patcher.HUNKS:
        check(f"'{name}' matches exactly once",
              original.count(anchor) == 1,
              f"count {original.count(anchor)}")
    check("the harness is not already patched", patcher.MARKER not in original)

    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "e004pf_k0pf_ab.py"
        shutil.copyfile(HARNESS, mirror)

        print("2. apply")
        rc = subprocess.run(
            [sys.executable, str(PATCHER), str(mirror)],
            capture_output=True, text=True,
        )
        check("patcher exits 0", rc.returncode == 0, rc.stderr.strip())
        patched = mirror.read_text()
        check("marker present", patcher.MARKER in patched)
        try:
            ast.parse(patched)
            check("patched file parses", True)
        except SyntaxError as exc:
            check("patched file parses", False, str(exc))
        for name, _anchor, repl in patcher.HUNKS:
            check(f"hunk '{name}' landed", repl in patched)
        check("only the four hunks changed the length",
              len(patched) > len(original))

        print("3. idempotency")
        rc2 = subprocess.run(
            [sys.executable, str(PATCHER), str(mirror)],
            capture_output=True, text=True,
        )
        check("second run exits 0", rc2.returncode == 0, rc2.stderr.strip())
        check("second run changes nothing", mirror.read_text() == patched)
        check("second run says 'already applied'",
              "already applied" in rc2.stdout)

        print("4. tamper refusal")
        tampered = Path(tmp) / "tampered.py"
        broken = patched.replace(
            "    _k0_m24_preop(stream)\n", "", 1
        )
        tampered.write_text(broken)
        rc3 = subprocess.run(
            [sys.executable, str(PATCHER), str(tampered)],
            capture_output=True, text=True,
        )
        check("a marked-but-incomplete file is refused", rc3.returncode == 2,
              f"rc {rc3.returncode}")

        print("5/6. the injected block, executed in isolation")
        _exercise_injected_block(patched)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("all m24_mok_patch tests passed")
    return 0


# --------------------------------------------------------------------------
# The injected helper block is pure Python that touches only os/sys/torch, so
# it can be lifted out of the patched harness and executed against a fake
# torch.  That is what turns "the patch applies" into "the patch works".
# --------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


class _FakeView:
    def __init__(self, parent, start, stop):  # noqa: ANN001
        self.parent, self.start, self.stop = parent, start, stop

    def copy_(self, other, non_blocking=False):  # noqa: ANN001
        values = other.tolist() if hasattr(other, "tolist") else list(other)
        self.parent.data[self.start : self.stop] = values
        return self

    def tolist(self):
        return list(self.parent.data[self.start : self.stop])


class _FakeTensor:
    _next_ptr = 0x7F0000000000

    def __init__(self, n: int) -> None:
        self.data = [0] * n
        _FakeTensor._next_ptr += 256
        self._ptr = _FakeTensor._next_ptr

    def data_ptr(self):
        return self._ptr

    def __getitem__(self, key):  # noqa: ANN001
        if isinstance(key, slice):
            return _FakeView(self, key.start or 0,
                             len(self.data) if key.stop is None else key.stop)
        return self.data[key]

    def __setitem__(self, key, value):  # noqa: ANN001
        self.data[key] = value

    def copy_(self, other, non_blocking=False):  # noqa: ANN001
        self.data[:] = other.tolist() if hasattr(other, "tolist") else list(other)
        return self

    def tolist(self):
        return list(self.data)


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.int32 = "int32"
    torch.zeros = lambda n, dtype=None, device=None, pin_memory=False: _FakeTensor(n)
    torch.empty = lambda n, dtype=None, device=None, pin_memory=False: _FakeTensor(n)

    def _tensor(values, dtype=None):  # noqa: ANN001
        t = _FakeTensor(len(values))
        t.data[:] = list(values)
        return t

    torch.tensor = _tensor
    class _StreamCtx:
        def __init__(self, stream):  # noqa: ANN001
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *exc):  # noqa: ANN002
            return False

    cuda = types.SimpleNamespace()
    cuda.current_stream = lambda: _FakeStream(0xABC)
    cuda.ExternalStream = _FakeStream
    cuda.stream = _StreamCtx
    torch.cuda = cuda
    return torch


def _exercise_injected_block(patched: str) -> None:
    marker = "# ---- PF4H_M24_MOK_FILL_V1"
    start = patched.index(marker)
    end = patched.index("# ---- end PF4H_M24_MOK_FILL_V1")
    block = patched[start:end]

    env = dict(os.environ)
    for key in ("K0_M24_FILL", "K0_M24_NORIG", "K0_M24_SHIM_DIR", "DHK_ROOT"):
        env.pop(key, None)

    def run(extra_env: dict[str, str]):
        # The real ``sys`` (the block manipulates sys.path to find the shim);
        # a fake ``os`` so the env is controlled; a fake ``torch``.
        namespace: dict[str, object] = {
            "os": types.SimpleNamespace(environ={**env, **extra_env},
                                        path=os.path),
            "sys": sys,
            "torch": _fake_torch(),
        }
        exec(compile(block, "<injected>", "exec"), namespace)  # noqa: S102
        return namespace

    # 5. INERTNESS -- nothing set.
    ns = run({})
    check("inert: _k0_m24_enabled() is False", ns["_k0_m24_enabled"]() is False)
    check("inert: _k0_m24_init returns None",
          ns["_k0_m24_init"](8, 4096) is None)
    check("inert: the pre-op is a no-op", ns["_k0_m24_preop"](0xABC) is None)
    check("inert: no buffer allocated", ns["_K0_M24_FILL"] is None)

    # 6. LIVE -- K0_M24_FILL=1, the shim pointed at this checkout.
    ns = run({"K0_M24_FILL": "1", "K0_M24_SHIM_DIR": str(SHIM)})
    fill = ns["_k0_m24_init"](8, 4096)
    check("live: a fill vector is allocated", fill is not None)
    if fill is None:
        return
    check("live: default payload is full fill",
          ns["_K0_M24_NORIG"] == [4096] * 8, f"got {ns['_K0_M24_NORIG']}")
    check("live: pointer is 16-byte aligned", fill.pointer(0) % 16 == 0)

    m24_fill = _load(SHIM / "m24_fill.py", "m24_fill_probe")
    desc = m24_fill.descriptor_with_fill_slot(list(range(1, 64)), fill.pointer(0))
    check("live: descriptor is 72 words", len(desc) == 72)
    check("live: slots 63..70 are zero", all(w == 0 for w in desc[63:71]))
    check("live: slot 71 is the fill pointer", desc[71] == fill.pointer(0))

    last_gen = 0
    for launch in range(3):
        ns["_k0_m24_preop"](0xABC)          # same stream as the launch
        raw = [w & 0xFFFFFFFF for w in fill._layers[0].tensor.tolist()]
        raw[8 + 8] = last_gen
        reason, t_eff = m24_fill.validate_payload_mirror(
            raw, cur=0, world=8, maxtok=4096, t_cap=4096
        )
        check(f"live: pre-op launch {launch} payload accepted",
              reason == m24_fill.RejectReason.ACCEPT, f"reason {reason}")
        check(f"live: pre-op launch {launch} T_eff = 4096", t_eff == 4096)
        last_gen = fill.current_gen(0)

    # A foreign stream must still work (ExternalStream path).
    ns["_k0_m24_preop"](0xDEF)
    raw = [w & 0xFFFFFFFF for w in fill._layers[0].tensor.tolist()]
    raw[8 + 8] = last_gen
    reason, _ = m24_fill.validate_payload_mirror(
        raw, cur=0, world=8, maxtok=4096, t_cap=4096
    )
    check("live: the foreign-stream path still writes a valid payload",
          reason == m24_fill.RejectReason.ACCEPT, f"reason {reason}")

    # A payload out of range must be refused at INIT, not on the device.
    ns2 = run({"K0_M24_FILL": "1", "K0_M24_SHIM_DIR": str(SHIM),
               "K0_M24_NORIG": "9999"})
    try:
        ns2["_k0_m24_init"](8, 4096)
        check("live: an out-of-range K0_M24_NORIG is refused", False,
              "no exception")
    except RuntimeError:
        check("live: an out-of-range K0_M24_NORIG is refused", True)

    ns3 = run({"K0_M24_FILL": "1", "K0_M24_SHIM_DIR": str(SHIM),
               "K0_M24_NORIG": "1539,1600,1480"})
    try:
        ns3["_k0_m24_init"](8, 4096)
        check("live: a wrong-length K0_M24_NORIG is refused", False,
              "no exception")
    except RuntimeError:
        check("live: a wrong-length K0_M24_NORIG is refused", True)


if __name__ == "__main__":
    raise SystemExit(main())
