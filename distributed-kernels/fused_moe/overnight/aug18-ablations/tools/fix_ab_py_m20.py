#!/usr/bin/env python3
"""Repair the anchor-A MoK harness so no-replication runs can start again.

WHAT IS BROKEN
--------------
`e004pf_k0pf_ab.py` binds `_m20` only *inside* the `if K0_MOK_REP_EXPERTS:`
block (introduced by ~/amd-master `14a42ca8`, 2026-08-14, "M20 slot-pool
harness").  The MPS descriptor builder later reads `_m20` unconditionally
(`if _m20 is not None:` and `_expected_mps_words = 71`).  With replication off
-- which is exactly the plain-M15 configuration that exp_03's C-ladder, R7 and
R4 all use -- `_m20` is never bound and all 8 ranks raise

    RuntimeError: PF6 prefill arm requires k0_n2 for the unchanged harness
                  controls: n2 buffer setup: NameError: name '_m20' is not defined

before touching a GPU.  The run dies in ~20 s with rc=23 ("did not produce
eight rank JSON files"), zero `[MARK]` lines and zero rank JSONs -- i.e. VOID
by HARNESS_MAP section 7's discriminator, NOT a hang.  Verified 2026-08-18 by
replaying the aug12 known-good invocation verbatim (C=16, g=353, mode=12,
flush_rows=16): it fails identically.  The sibling globals were done correctly
(`_m18_rep_buf = None` is a module-level default); `_m20` simply missed it.

THE FIX
-------
Insert one additive line -- `_m20 = None` -- immediately before the
`if K0_MOK_REP_EXPERTS:` guard, at the guard's own indent, i.e. in the same
scope where the guarded body already assigns it.  Replication-on behaviour is
bit-identical (the guard body reassigns `_m20` on its first statement).

WHY THIS CANNOT DISTURB R7's "SHIPPING BINARY, NO REBUILDS" GUARANTEE
--------------------------------------------------------------------
`ab.py` is a host script.  The mori JIT cache key hashes only files with
suffix .hpp/.h/.cpp/.hip under the mori source dirs (HARNESS_MAP section 3),
so this edit cannot invalidate `k0pf6gm_mps_mega.hsaco`.  The kernel that R7
measures is byte-identical before and after; `summary.json ->
kernel_hsaco_sha256` is the receipt.

USAGE (run ON the node; never edits anything but the one target file)
---------------------------------------------------------------------
    python3 fix_ab_py_m20.py --check    # report state, write nothing
    python3 fix_ab_py_m20.py --apply    # back up, patch, AST-validate
    python3 fix_ab_py_m20.py --revert   # restore the backup

Idempotent: --apply on an already-patched file is a no-op that exits 0.
"""

import argparse
import ast
import os
import shutil
import sys

DEFAULT_TARGET = os.path.expanduser(
    "~/amd-master/auto-gpu-kernel/k0_fused_moe/prefill_opt/host/e004pf_k0pf_ab.py"
)
BACKUP_SUFFIX = ".pre_m20_noreplication_fix"

GUARD = "    if K0_MOK_REP_EXPERTS:"
GUARDED_ASSIGN = "        _m20 = None"
FIX_LINE = (
    "    _m20 = None  # no-replication path: M20 state must be bound before "
    "the K0_MOK_REP_EXPERTS guard (see aug18-ablations/tools/fix_ab_py_m20.py)"
)
FIX_MARKER = "no-replication path: M20 state must be bound"


def locate(lines):
    """Return (guard_idx, guarded_assign_idx) or raise with a clear message."""
    assigns = [i for i, l in enumerate(lines) if l == GUARDED_ASSIGN]
    if len(assigns) != 1:
        raise SystemExit(
            f"REFUSING: expected exactly one {GUARDED_ASSIGN!r} line, found "
            f"{len(assigns)} at {[i + 1 for i in assigns]}. The file has drifted; "
            "re-read it before patching."
        )
    j = assigns[0]
    guards = [i for i, l in enumerate(lines[:j]) if l == GUARD]
    if not guards:
        raise SystemExit(
            f"REFUSING: no {GUARD!r} line found above line {j + 1}. "
            "The file has drifted; re-read it before patching."
        )
    return guards[-1], j


def already_patched(lines, guard_idx):
    window = lines[max(0, guard_idx - 3):guard_idx]
    return any(FIX_MARKER in l for l in window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = os.path.expanduser(args.target)
    backup = target + BACKUP_SUFFIX

    if args.revert:
        if not os.path.exists(backup):
            raise SystemExit(f"no backup at {backup}")
        shutil.copyfile(backup, target)
        print(f"REVERTED {target} from {backup}")
        return 0

    src = open(target).read()
    lines = src.split("\n")
    guard_idx, assign_idx = locate(lines)

    print(f"target        : {target}")
    print(f"guard         : line {guard_idx + 1}  {GUARD!r}")
    print(f"guarded assign: line {assign_idx + 1}  {GUARDED_ASSIGN!r}")

    if already_patched(lines, guard_idx):
        print("STATE: already patched -- nothing to do.")
        return 0
    print("STATE: UNPATCHED (no-replication runs will die with the _m20 NameError)")

    if args.check:
        print("\n--check: no files written. Re-run with --apply to patch.")
        return 0

    lines.insert(guard_idx, FIX_LINE)
    patched = "\n".join(lines)

    # Never write a file that will not parse.
    ast.parse(patched)
    print("AST: patched source parses cleanly")

    if not os.path.exists(backup):
        shutil.copyfile(target, backup)
        print(f"backup written: {backup}")
    else:
        print(f"backup already exists, left as-is: {backup}")

    with open(target, "w") as fh:
        fh.write(patched)

    print(f"\nPATCHED: inserted at line {guard_idx + 1}:")
    for i in range(max(0, guard_idx - 1), min(len(lines), guard_idx + 3)):
        mark = ">>" if i == guard_idx else "  "
        print(f"{mark} {i + 1}: {lines[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
