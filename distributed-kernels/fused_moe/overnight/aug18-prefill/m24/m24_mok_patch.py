#!/usr/bin/env python3
"""Wire the M24 fill vector into the MoK harness.  Anchor-exact, idempotent.

WHY THIS EXISTS.  ``K0P6_M24_FILL=1`` makes the megakernel read descriptor
slot 71, dereference it, and STORE through it.  The MoK harness builds a
63..71-word descriptor and knows nothing about slot 71, so before this patch
every fill arm read 72 bytes past the descriptor tensor and, whenever the
residue happened to look like a non-null 16-byte-aligned address, wrote through
it on all 8 GPUs.  Both adversarial reviews returned this as blocking.

It also closes the second half of the same hole.  The rev-3 kernel no longer
short-circuits its validation under ``NORIG_CONST`` / ``NORIG_TABLE`` -- the
MoK binary now runs the SAME validate/reject/publish path as the serving
binary, which is the entire point of the section-D "the MoK binary is the
shipping binary" box.  That means a MoK arm needs a genuinely valid, genuinely
fresh payload before every launch, exactly as serving does.  Without it every
step would reject with ``RJ_STALE`` after the first, fall back to the padded
capacity, and the arm would silently measure the pre-M24 kernel.

WHAT IT CHANGES (four hunks, all no-ops unless ``K0_M24_FILL`` is set):

  1. a helper block after ``sp()``: lazily imports ``m24_fill`` from the DHK
     checkout and allocates the per-layer buffer;
  2. the MPS descriptor build: extends the cascade descriptor to 72 words with
     slot 71 bound and every unclaimed slot zeroed;
  3. ``_expected_mps_words``: 72 on a fill build, so the harness's own drift
     assertion still fires on a mis-built descriptor;
  4. ``_pf6mps_mega``: the on-stream pre-op, issued on the SAME stream as the
     launch that follows it.

ENV:
  ``K0_M24_FILL=1``        turn the plumbing on (must match the pin's macros)
  ``K0_M24_NORIG=a,b,...`` the per-rank n_orig payload (default: MAXTOK on
                           every rank, i.e. full fill -- the value the
                           NORIG_CONST/NORIG_TABLE pins override on-device)
  ``K0_M24_SHIM_DIR``      where ``m24_fill.py`` lives (default: derived from
                           ``DHK_ROOT``)

Usage:
    python3 m24_mok_patch.py [--check] [PATH_TO_e004pf_k0pf_ab.py]

Exit codes: 0 applied or already applied, 2 anchor mismatch / refusal.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

MARKER = "PF4H_M24_MOK_FILL_V1"

DEFAULT_TARGET = (
    "~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/"
    "e004pf_k0pf_ab.py"
)

# --------------------------------------------------------------------------
# Hunk 1 -- the helper block, appended after sp().
# --------------------------------------------------------------------------
ANCHOR_SP = "def sp(): return torch.cuda.current_stream().cuda_stream"

BLOCK_HELPER = f'''

# ---- {MARKER} ------------------------------------------------------------
# M24 fill vector for the MoK arms.  Inert unless K0_M24_FILL is set, and the
# ONLY thing it does when set is give the kernel a valid payload to validate --
# the fill EFFECT comes from the pin's K0P6_M24_NORIG_* macros, never from
# here, so this block is identical across the whole ladder.
_K0_M24_FILL = None
_K0_M24_NORIG = None


def _k0_m24_enabled():
    return os.environ.get("K0_M24_FILL", "").strip() not in ("", "0")


def _k0_m24_init(world, maxtok):
    """Allocate the per-layer fill buffer.  Returns None when M24 is off."""
    global _K0_M24_FILL, _K0_M24_NORIG
    if not _k0_m24_enabled():
        return None
    if _K0_M24_FILL is not None:
        return _K0_M24_FILL
    shim = os.environ.get("K0_M24_SHIM_DIR", "").strip()
    if not shim:
        root = os.environ.get("DHK_ROOT", "").strip()
        if not root:
            raise RuntimeError(
                "K0_M24_FILL is set but neither K0_M24_SHIM_DIR nor DHK_ROOT "
                "is; the pin discipline requires an explicit DHK_ROOT anyway "
                "(DECOMP_RUNBOOK 4.2)"
            )
        shim = os.path.join(
            os.path.expanduser(root),
            "distributed-kernels/fused_moe/overnight/aug18-prefill/m15_eplb0",
        )
    if shim not in sys.path:
        sys.path.insert(0, shim)
    import m24_fill as _m24
    raw = os.environ.get("K0_M24_NORIG", "").strip()
    if raw:
        vals = [int(x) for x in raw.replace(",", " ").split()]
        if len(vals) == 1:
            vals = vals * int(world)
    else:
        # Full fill.  The device-side substitution pins are what move the
        # ladder; the harness payload only has to be VALID.
        vals = [int(maxtok)] * int(world)
    if len(vals) != int(world):
        raise RuntimeError(
            "K0_M24_NORIG has %d entries, world is %d" % (len(vals), world)
        )
    if any(v < 0 or v > int(maxtok) for v in vals):
        raise RuntimeError(
            "K0_M24_NORIG entries must be in [0, MAXTOK=%d]; got %r"
            % (maxtok, vals)
        )
    _K0_M24_NORIG = vals
    _K0_M24_FILL = _m24.M24FillVector(
        world=int(world), num_layers=1, maxtok=int(maxtok),
        torch_module=torch,
    )
    print("[M24] fill vector at %#x, n_orig=%r" % (
        _K0_M24_FILL.pointer(0), vals), flush=True)
    return _K0_M24_FILL


def _k0_m24_preop(stream):
    """One step's payload, ON THE LAUNCH'S STREAM.  ~2 us, inside timing."""
    if _K0_M24_FILL is None:
        return
    cur = torch.cuda.current_stream()
    if int(cur.cuda_stream) == int(stream):
        _K0_M24_FILL.write(0, _K0_M24_NORIG)
    else:
        with torch.cuda.stream(torch.cuda.ExternalStream(int(stream))):
            _K0_M24_FILL.write(0, _K0_M24_NORIG)
# ---- end {MARKER} --------------------------------------------------------
'''

# --------------------------------------------------------------------------
# Hunk 2 -- extend the MPS descriptor before it becomes a tensor.
# --------------------------------------------------------------------------
ANCHOR_DESC = (
    '                pf6_state["desc_mps"] = torch.tensor(\n'
    "                    _pf6mps_desc_list, dtype=torch.int64, device=\"cuda\"\n"
    "                )"
)

BLOCK_DESC = (
    f"                # {MARKER}: bind descriptor slot 71 and zero-fill every\n"
    "                # slot this build's cascade does not claim.  Without this\n"
    "                # the kernel's unconditional desc[71] read is out of\n"
    "                # bounds of the descriptor tensor.\n"
    "                if _k0_m24_init(WORLD, MAXTOK) is not None:\n"
    "                    import m24_fill as _m24mod\n"
    "                    _pf6mps_desc_list = _m24mod.descriptor_with_fill_slot(\n"
    "                        _pf6mps_desc_list, _K0_M24_FILL.pointer(0)\n"
    "                    )\n"
) + ANCHOR_DESC

# --------------------------------------------------------------------------
# Hunk 3 -- the harness's own descriptor-length assertion.
# --------------------------------------------------------------------------
ANCHOR_WORDS = (
    "                if _m20 is not None:\n"
    "                    _expected_mps_words = 71"
)

BLOCK_WORDS = ANCHOR_WORDS + (
    f"\n                # {MARKER}: M24 raises K0P6_M15_D_LEN to 72 from\n"
    "                # whatever the cascade produced.\n"
    "                if _K0_M24_FILL is not None:\n"
    "                    _expected_mps_words = 72"
)

# --------------------------------------------------------------------------
# Hunk 4 -- the on-stream pre-op, immediately before the mega launch.
# --------------------------------------------------------------------------
ANCHOR_LAUNCH = (
    '    if not os.environ.get("K0_MPS_SKIP_LAUNCH"):\n'
    "        fn_pf6mps_mega.launch("
)

BLOCK_LAUNCH = (
    f"    # {MARKER}: the fill vector is a per-STEP payload with a\n"
    "    # consume-once gen, so it is refreshed before every launch -- a graph\n"
    "    # replay on a stale payload is exactly what RJ_STALE rejects.\n"
    "    _k0_m24_preop(stream)\n"
) + ANCHOR_LAUNCH


HUNKS = (
    ("helper block after sp()", ANCHOR_SP, ANCHOR_SP + BLOCK_HELPER),
    ("MPS descriptor slot 71", ANCHOR_DESC, BLOCK_DESC),
    ("_expected_mps_words = 72", ANCHOR_WORDS, BLOCK_WORDS),
    ("on-stream pre-op", ANCHOR_LAUNCH, BLOCK_LAUNCH),
)


def apply(path: Path, *, check_only: bool = False) -> int:
    if not path.is_file():
        print(f"[m24_mok_patch] REFUSE: {path} is not a file", file=sys.stderr)
        return 2
    source = path.read_text()

    if MARKER in source:
        # Idempotent: prove all four hunks are present, not just the marker.
        missing = [
            name
            for name, _anchor, replacement in HUNKS
            if replacement not in source
        ]
        if missing:
            print(
                "[m24_mok_patch] REFUSE: the marker is present but these "
                f"hunks are not: {missing}. The file was hand-edited; "
                "restore it before re-patching.",
                file=sys.stderr,
            )
            return 2
        print("[m24_mok_patch] already applied (all 4 hunks present)")
        return 0

    for name, anchor, _replacement in HUNKS:
        count = source.count(anchor)
        if count != 1:
            print(
                f"[m24_mok_patch] REFUSE: anchor for '{name}' matched {count} "
                "times, expected exactly 1. The harness moved; re-derive the "
                "anchor rather than loosening it.",
                file=sys.stderr,
            )
            return 2

    if check_only:
        print("[m24_mok_patch] all 4 anchors match exactly once (not applied)")
        return 0

    patched = source
    for _name, anchor, replacement in HUNKS:
        patched = patched.replace(anchor, replacement, 1)

    try:
        ast.parse(patched)
    except SyntaxError as exc:  # pragma: no cover - a patch bug, not a run bug
        print(
            f"[m24_mok_patch] REFUSE: the patched file does not parse: {exc}",
            file=sys.stderr,
        )
        return 2

    path.write_text(patched)
    print(f"[m24_mok_patch] applied 4 hunks to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default=DEFAULT_TARGET)
    parser.add_argument(
        "--check", action="store_true",
        help="verify the anchors without writing",
    )
    args = parser.parse_args(argv)
    return apply(
        Path(os.path.expanduser(args.target)), check_only=args.check
    )


if __name__ == "__main__":
    raise SystemExit(main())
