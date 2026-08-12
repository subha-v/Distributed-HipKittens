#!/usr/bin/env python3
"""exp_24 re-measure check A: does the binary under test really group at 2 on
shape 5?

Host arithmetic only -- `dhk_rt.resolve_shape` is the same shape planner the
kernel launch uses, so the tiles/producers it reports ARE the numbers the
device sees. This is exp_26's `ab_pershape.py` check (`expected_rgroup` per arm
plus `arms._geometry` with tiles / producers / tiles_per_cta), applied to the
single shipped configuration instead of five arms.

Lives in a file rather than a heredoc on purpose: the first attempt piped this
through `python3 -` over the remote shell, the heredoc did not survive, and
python exited 0 on empty stdin -- a silent false pass on the exact assertion
this script exists to make.
"""
import importlib.util
import json
import sys
import os

D = "/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders"
RT = ("/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/harness/"
      "build/dhk_rt.so")

# The graded table is IMPORTED from the measurement script, never retyped here.
# The first attempt hardcoded it and got rows 3 and 4 wrong (2048x4096x12288
# instead of 2048x2880x2880, and every bias flag inverted); both fell through to
# the generic config row 0, which reports 4096 tiles at 15 per CTA and made the
# check fail on the kernel's behalf. A verification that can disagree with the
# thing it verifies is not a verification.
sys.path.insert(0, D)
import ladder_mp  # noqa: E402

SCORED = [(m, n, k, bias) for m, n, k, bias, _seed in ladder_mp.SCORED]
RELEASE_GROUP = 4
EXPECT_NOW = [1, 1, 1, 1, 2, 4]
EXPECT_WAS = [1, 1, 1, 1, 1, 4]


def incumbent_rgroup(ppc):
    """PERSHAPE=0: the two-element set {RELEASE_GROUP, 1}."""
    return RELEASE_GROUP if ppc >= RELEASE_GROUP else 1


def shipped_rgroup(ppc):
    """PERSHAPE=2: descending select over the literal ladder {RG, 2, 1}."""
    if ppc >= RELEASE_GROUP:
        return RELEASE_GROUP
    if ppc >= 2:
        return 2
    return 1


def main():
    spec = importlib.util.spec_from_file_location("dhk_rt", RT)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)

    rows, ok, moved = [], True, []
    print(f"  {'shape':>20} {'row':>4} {'tiles':>6} {'prod':>5} {'t/CTA':>6} "
          f"{'was':>4} {'now':>4}  note")
    for i, (m, n, k, bias) in enumerate(SCORED):
        p = rt.resolve_shape(m, n, k, bias)
        tiles = int(p["gemm_tiles"])
        prod = int(p["num_gemm_ctas"])
        ppc = -(-tiles // prod)
        was, now = incumbent_rgroup(ppc), shipped_rgroup(ppc)
        note = ""
        if was != now:
            moved.append(i + 1)
            note = "<== the row exp_26 moves"
        if now != EXPECT_NOW[i]:
            note = f"FATAL expected {EXPECT_NOW[i]}"
            ok = False
        if was != EXPECT_WAS[i]:
            note += f" FATAL prev expected {EXPECT_WAS[i]}"
            ok = False
        print(f"  {m}x{n}x{k:<7} {int(p['config_row']):>4} {tiles:>6} "
              f"{prod:>5} {ppc:>6} {was:>4} {now:>4}  {note}")
        rows.append({"shape_index": i, "shape": [m, n, k, int(bias)],
                     "config_row": int(p["config_row"]), "tiles": tiles,
                     "producers": prod, "tiles_per_cta": ppc,
                     "rgroup_prev": was, "rgroup_now": now,
                     "num_reducer_ctas": int(p["num_reducer_ctas"])})

    print(f"\n  rgroup ladder now  = {[r['rgroup_now'] for r in rows]}")
    print(f"  rgroup ladder prev = {[r['rgroup_prev'] for r in rows]}")
    print(f"  rows that move     = {moved}")
    if moved != [5]:
        print(f"  FATAL: exactly shape 5 must move, got {moved}")
        ok = False
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    with open(sys.argv[1] if len(sys.argv) > 1 else "geometry.json", "w") as fh:
        json.dump({"release_group": RELEASE_GROUP, "pershape": 2,
                   "rows": rows, "moved": moved, "pass": ok}, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
