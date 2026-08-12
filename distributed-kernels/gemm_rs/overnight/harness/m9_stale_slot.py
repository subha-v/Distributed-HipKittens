"""Gate M9: stale-slot detector -- poisoned heap, bitwise golden, publish-early.

All three M4 controls are blind to the one property a batched release puts at
risk: that no `ready = e` becomes visible before the payload bytes it describes.
A build that coarsens the release while leaving publication per tile would let a
reducer acquire a slot still dirty in the producer's L2, and its buffer_inv
would then pull the PREVIOUS epoch's line -- which is the correct answer
whenever the inputs did not change. That is exactly the blindness the ledger
already records for CTRL_REROUTE_SLOT, and no tolerance can repair it: one wrong
1-of-8 contribution lands at ~7e-3, inside the graded 1e-2.

Four ingredients, all outside any timed region:

  (a) POISON. Every rank's c_heap and out are filled with bf16 quiet NaN before
      every launch. If the protocol is correct this is unobservable -- every
      slot a reducer reads at epoch e was fully rewritten by its producer before
      ready = e was published. A stale read becomes NaN instead of a plausible
      number, so detection needs no tolerance argument at all. The golden arm
      runs under the same poison, so a NaN there would indict the poison rather
      than the candidate.
  (b) BITWISE GOLDEN. This kernel is bit-exact deterministic: fixed
      source-ascending fp32 reduction with a single RNE pack, no atomics in the
      arithmetic path, fixed tile->CTA map. A release-granularity change must
      therefore not move a single bit. Asserted with torch.equal against a
      frozen pre-E3 build driven over the same input sequence in the same
      process -- orders of magnitude stronger than 2e-3, and free.
  (c) STRESS. Many epochs, changing inputs, rotated launch skew, on the shapes
      that actually batch: 8192x8192x29568 (1024 tiles over 272 producers = 4
      tiles per CTA with a 3-tile tail) and the generic row at 8192x8192x28672
      (32768 tiles over 280 producers = up to 118 tiles per CTA), which the
      scored table never reaches and which is the case that makes an unbounded
      group rule unacceptable.
  (d) CTRL_PUBLISH_EARLY. The control of the control. A build that publishes the
      group BEFORE releasing it must be caught by (a)+(b)+(c). If it is not,
      this gate has no power over the property it exists to test and no batched
      arm may be called correct on its evidence.

Usage:
  python3 m9_stale_slot.py            # full gate
  python3 m9_stale_slot.py 0.1        # scale every epoch count (smoke)
"""

import math
import sys
import time

import torch

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

TIGHT = 2e-3

# The golden kernel, built as a separate module by
# experiments/exp_05_release_granularity/build_golden.sh.
#
# THE GOLDEN IS NOT PERMANENTLY FROZEN, and treating it as if it were is how
# this gate stopped working. It must be REBUILT from the immediately preceding
# validated source whenever the shape table changes, for two independent
# reasons:
#
#   1. A bitwise comparison across a BM/BN/BK change is meaningless. Different
#      tiling changes the accumulation order, so the bits legitimately differ
#      and the gate reports corruption that is not there.
#   2. A golden carrying an older table will compute a DIFFERENT tile map than
#      the plan the harness hands it. That is not a mismatch, it is an
#      out-of-bounds access: the pre-exp_14 golden reads 192 rows past the end
#      of a 2880-row B operand on row 3 and faults the GPU
#      ("Memory access fault by GPU node-7"), which then wedges the node in
#      driver teardown and blocks every other experiment.
#
# The staleness check below turns that fault into a clear refusal. Regenerate
# with experiments/exp_05_release_granularity/build_golden.sh against the
# current best source before running this gate after any table change.
GOLDEN_MODULE = "gemm_rs_mi300x_e3base"

# Written by build_golden.sh next to the golden module: the scored-shape rows
# the golden was compiled from. Absent on goldens built before this guard.
GOLDEN_TABLE_SIDECAR = "build/gemm_rs_mi300x_e3base.table.json"


def _assert_golden_matches_current_table(cases):
    """Refuse to run if the golden was built from a different shape table.

    Cheap, and it converts a GPU memory fault that wedges the node into a
    one-line message naming the row that moved.
    """
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, GOLDEN_TABLE_SIDECAR)
    current = {}
    for (m, n, k, bias), _epochs, _why in cases:
        p = rt.resolve_shape(m, n, k, bias)
        current[f"{m}x{n}x{k}x{int(bias)}"] = [p["bm"], p["bn"], p["bk"]]

    if not os.path.exists(path):
        raise SystemExit(
            f"GATE M9 REFUSED: no golden table sidecar at {path}.\n"
            f"  The golden module '{GOLDEN_MODULE}' cannot be shown to share the\n"
            f"  current shape table, and a golden built from a different table\n"
            f"  reads out of bounds and faults the GPU rather than failing.\n"
            f"  Rebuild it: experiments/exp_05_release_granularity/build_golden.sh\n"
            f"  Current table for this gate's cases: {current}")

    with open(path) as handle:
        built_from = json.load(handle)

    drift = {k: (built_from.get(k), v) for k, v in current.items()
             if built_from.get(k) != v}
    if drift:
        lines = "\n".join(f"    {k}: golden {g} vs current {c}"
                          for k, (g, c) in drift.items())
        raise SystemExit(
            f"GATE M9 REFUSED: the golden predates the current shape table.\n"
            f"{lines}\n"
            f"  A bitwise comparison across a tile change is meaningless (the\n"
            f"  accumulation order differs), and the older tile map reads out of\n"
            f"  bounds. Rebuild the golden from the current best source with\n"
            f"  experiments/exp_05_release_granularity/build_golden.sh")

# hipMemset carries a single byte, so the poison pattern has to be byte-uniform.
# bf16 0xFFFF is sign 1, exponent all ones, mantissa 0x7F with its leading bit
# set: a negative quiet NaN. The canonical 0x7FC0 is not byte-uniform.
POISON_BYTE = 0xFF

# Mirrors hk_gemm_rs_mi300x::CTRL_PUBLISH_EARLY in gemm_rs_mi300x_constants.cuh.
# dhk_rt re-exports the original three control bits only.
CTRL_PUBLISH_EARLY = getattr(rt, "CTRL_PUBLISH_EARLY", 1 << 3)

# (m, n, k, bias), epochs, why this shape is here. Epoch floors are the protocol
# review's; the last two shapes are the only ones that batch more than 2 tiles.
CASES = [
    ((512, 4096, 12288, True), 600, "row 2, 2 tiles/CTA with a 1-tile tail"),
    ((8192, 8192, 29568, False), 60, "row 6, 4 tiles/CTA with a 3-tile tail"),
    ((8192, 8192, 28672, False), 20, "generic row, up to 118 tiles/CTA"),
    # The one-tile-per-CTA shapes. No release granularity can group anything on
    # these, so the protocol review's condition C8 is that they come out
    # bit-identical -- which is a claim about output, and therefore a claim this
    # gate is the right place to test rather than to assert. Few epochs: they are
    # here for the bitwise comparison, not for the race.
    ((64, 7168, 18432, False), 30, "row 1, 1 tile/CTA, 216 idle CTAs (C8)"),
    ((2048, 2880, 2880, True), 30, "row 3, 1 tile/CTA, 80 idle CTAs (C8)"),
    ((4096, 4096, 4096, False), 30, "row 4, 1 tile/CTA, 16 idle CTAs (C8)"),
]

# The publish-early control needs enough epochs to race, not a full soak. The
# one-tile shapes still get a few: publishing before the release is wrong there
# too, it just cannot be made wrong by BATCHING.
CONTROL_EPOCHS = [20, 20, 10, 10, 10, 10]

failures = []
notes = []


def check(condition, message):
    print(f"    {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        failures.append(message)
    return condition


def poison(h):
    """bf16 NaN over every slot of every rank's heap, and over every output."""
    nbytes = int(h.plan["c_heap_bytes"])
    for rank in range(WORLD):
        rt.fill_bytes(rank, h.c_heap[rank], POISON_BYTE, nbytes)
        torch.cuda.set_device(rank)
        h.out[rank].fill_(float("nan"))
    h.sync()


def nan_ranks(h):
    out = []
    for rank in range(WORLD):
        count = int(torch.isnan(h.out[rank]).sum())
        if count:
            out.append((rank, count, int(h.out[rank].numel())))
    return out


def tiles_per_cta(plan):
    tiles = int(plan["gemm_tiles"])
    producers = int(plan["num_gemm_ctas"])
    return tiles, producers, -(-tiles // producers)


def sweep(shape, epochs, why, label, control=False, ctrl_flags=0):
    """Run `epochs` poisoned, skewed, changing-input epochs of one shape.

    Returns a dict of how many epochs each detector fired on. The caller decides
    whether firing is a pass (the publish-early control) or a failure (a real
    candidate); this function never decides that, so both arms are measured by
    exactly the same instrument.
    """
    m, n, k, bias = shape
    plan = rt.resolve_shape(m, n, k, bias)
    tiles, producers, per_cta = tiles_per_cta(plan)
    print(f"\n-- {label}: m={m} n={n} k={k} bias={int(bias)} --")
    print(f"   {why}: {tiles} tiles over {producers} producer CTAs "
          f"= up to {per_cta} tiles per CTA; {epochs} epochs")

    fired = {"nan": [], "bitwise": [], "tight": [], "errbits": [],
             "golden_nan": [], "golden_errbits": []}
    worst = 0.0
    started = time.perf_counter()
    gold = GemmRS(m, n, k, bias, module_name=GOLDEN_MODULE)
    cand = GemmRS(m, n, k, bias, control=control)
    try:
        for epoch in range(1, epochs + 1):
            seed = 5000 + epoch
            # Identical inputs by construction: generate_input reseeds a private
            # generator with (seed + rank), so both instances see the same bytes.
            gold.set_inputs(seed)
            cand.set_inputs(seed)
            skew = [(epoch + r) % WORLD for r in range(WORLD)]

            poison(gold)
            gold.launch(skew=skew)
            poison(cand)
            cand.launch(skew=skew, ctrl_flags=ctrl_flags)

            if gold.error_report():
                fired["golden_errbits"].append(epoch)
            if nan_ranks(gold):
                fired["golden_nan"].append(epoch)
            if cand.error_report():
                fired["errbits"].append(epoch)
            if nan_ranks(cand):
                fired["nan"].append(epoch)
            differing = [r for r in range(WORLD)
                         if not torch.equal(cand.out[r], gold.out[r])]
            if differing:
                fired["bitwise"].append(epoch)
            checks = cand.verify(rtol=TIGHT, atol=TIGHT)
            finite = [c["max_abs_diff"] for c in checks
                      if not math.isnan(c["max_abs_diff"])]
            worst = max([worst] + finite)
            if not all(c["allclose"] for c in checks):
                fired["tight"].append(epoch)

            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                elapsed = time.perf_counter() - started
                print(f"     epoch {epoch:>4}: nan={len(fired['nan'])} "
                      f"bitwise={len(fired['bitwise'])} "
                      f"tight={len(fired['tight'])} "
                      f"errbits={len(fired['errbits'])}  "
                      f"worst|diff|={worst:.3e}  {elapsed:6.1f}s")

        fired["signals"] = cand.check_signals()
        fired["epochs"] = cand.check_epochs()
    finally:
        cand.close()
        gold.close()
    fired["worst"] = worst
    fired["n"] = epochs
    return fired


def detected(fired):
    return bool(fired["nan"] or fired["bitwise"] or fired["tight"]
                or fired["errbits"])


def main():
    # Before any GPU work: a golden built from a different shape table does not
    # merely disagree, it reads out of bounds and faults the node.
    _assert_golden_matches_current_table(CASES)

    rt.enable_peer_access(WORLD)
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

    print("=" * 78)
    print("GATE M9 - stale-slot detector (poisoned heap + bitwise golden)")
    print(f"  golden module : {GOLDEN_MODULE} (table-matched, see sidecar)")
    print(f"  poison        : 0x{POISON_BYTE:02X} bytes -> bf16 0xFFFF (qNaN)")
    print(f"  epoch scale   : {scale}")
    print("=" * 78)

    # ---- the candidate must be invisible under the detector ----------------
    print("\n### candidate (production module) vs frozen pre-E3 golden ###")
    for (shape, epochs, why) in CASES:
        n = max(1, int(round(epochs * scale)))
        fired = sweep(shape, n, why, "candidate")
        check(not fired["golden_nan"] and not fired["golden_errbits"],
              f"golden itself is clean under the poison "
              f"(otherwise the poison, not the candidate, is wrong)")
        check(not fired["nan"],
              f"no NaN reached any output in {n} poisoned epochs "
              f"(stale-slot reads: {len(fired['nan'])})")
        check(not fired["bitwise"],
              f"bit-identical to the pre-E3 build on all 8 ranks for all {n} "
              f"epochs (differing epochs: {len(fired['bitwise'])})")
        check(not fired["tight"],
              f"allclose({TIGHT}) every epoch, worst |diff| "
              f"{fired['worst']:.3e}")
        check(not fired["errbits"], "no error bit was ever raised")
        check(not fired["signals"] and not fired["epochs"],
              "epoch cells and every touched ready/credit cell are exact")

    # ---- and the detector must have power over the inverted order ----------
    print("\n### CTRL_PUBLISH_EARLY (publication moved BEFORE the release) ###")
    print("    A gate that this control passes is theatre. It must fire.")
    any_fired = False
    for (shape, _, why), epochs in zip(CASES, CONTROL_EPOCHS):
        n = max(1, int(round(epochs * scale)))
        fired = sweep(shape, n, why, "CTRL_PUBLISH_EARLY", control=True,
                      ctrl_flags=CTRL_PUBLISH_EARLY)
        hit = detected(fired)
        any_fired = any_fired or hit
        print(f"    detector fired on: NaN {len(fired['nan'])}/{n} epochs, "
              f"bitwise {len(fired['bitwise'])}/{n}, "
              f"tight {len(fired['tight'])}/{n}, "
              f"errbits {len(fired['errbits'])}/{n}")
        if not hit:
            notes.append(
                f"CTRL_PUBLISH_EARLY did NOT fire on m={shape[0]} n={shape[1]} "
                f"k={shape[2]} over {n} epochs; the race window on this shape "
                f"is evidently too narrow to catch by sampling")
    check(any_fired,
          "CTRL_PUBLISH_EARLY is detected by this gate on at least one shape "
          "(the gate has power over release/publication ordering)")

    print("\n" + "=" * 78)
    for note in notes:
        print(f"note: {note}")
    if failures:
        print(f"GATE M9 FAILED: {len(failures)} expectation(s) unmet")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE M9 PASSED: no stale slot under a poisoned heap, bit-identical "
          "to the pre-E3 build, and the publish-early control does fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
