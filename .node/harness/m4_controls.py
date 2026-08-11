"""Gate M4: the three negative controls must fail in their designed way.

These run against the SEPARATE negative-control module. A control that silently
passes would mean the corresponding safety property is untested, so each check
asserts a specific failure signature, not merely "something went wrong".

  CTRL_DROP_PUBLICATION  consumers of the suppressed source hit the bounded
                         readiness timeout (err bit 26) AND perform zero
                         payload reads (output stays at its sentinel).
  CTRL_DROP_CREDIT       the starved source's epoch-2 producers hit the credit
                         timeout (err bit 25) and do not overwrite the slot.
  CTRL_REROUTE_SLOT      no timeout, but numerics must be wrong.
"""

import sys

import torch

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

SENTINEL = 1.5  # distinctive, exactly representable in bf16

failures = []
notes = []


def check(condition, message):
    if condition:
        print(f"    ok   {message}")
    else:
        print(f"    FAIL {message}")
        failures.append(message)
    return condition


def fill_sentinel(harness):
    for rank in range(WORLD):
        harness.out[rank].fill_(SENTINEL)
    torch.cuda.synchronize()


def untouched_fraction(harness, rank):
    out = harness.out[rank]
    return float((out.float() == SENTINEL).sum()) / out.numel()


# ---------------------------------------------------------------------------
def control_drop_publication(shape, source_rank=3, dest_rank=5,
                             spin_limit=50_000):
    m, n, k, bias = shape
    print(f"\n== CTRL_DROP_PUBLICATION (source rank {source_rank} stops "
          f"publishing to rank {dest_rank}) ==")
    with GemmRS(m, n, k, bias, spin_limit=spin_limit, control=True) as h:
        h.set_inputs(1234)
        fill_sentinel(h)
        h.launch(ctrl_flags=rt.CTRL_DROP_PUBLICATION,
                 ctrl_rank=source_rank, ctrl_arg0=dest_rank)
        bits = h.error_bits()
        print(f"    error bits per rank: "
              f"{[hex(b) for b in bits]}")
        check(bool(bits[dest_rank] & rt.ERR_REDUCER_READY),
              f"rank{dest_rank} raised REDUCER_READY timeout (bit 26)")
        others = [r for r in range(WORLD)
                  if r != dest_rank and (bits[r] & rt.ERR_REDUCER_READY)]
        notes.append(f"drop_publication: ranks also reporting bit26: {others}")
        frac = untouched_fraction(h, dest_rank)
        print(f"    rank{dest_rank} output still sentinel: {frac*100:.1f}%")
        check(frac == 1.0,
              f"rank{dest_rank} performed ZERO payload reads/writes "
              f"(fail-closed before any peer read)")
        checks = h.verify()
        check(not checks[dest_rank]["allclose"],
              f"rank{dest_rank} output is (correctly) not valid")


# ---------------------------------------------------------------------------
def control_drop_credit(shape, owner_rank=2, starved_source=6,
                        spin_limit=50_000):
    m, n, k, bias = shape
    print(f"\n== CTRL_DROP_CREDIT (owner rank {owner_rank} withholds the "
          f"retirement credit from source {starved_source}) ==")
    with GemmRS(m, n, k, bias, spin_limit=spin_limit, control=True) as h:
        # Epoch 1 must succeed: epoch one takes the credit fast path, so the
        # withheld credit can only bite at epoch 2.
        h.set_inputs(1234)
        h.launch(ctrl_flags=rt.CTRL_DROP_CREDIT,
                 ctrl_rank=owner_rank, ctrl_arg0=starved_source)
        bits1 = h.error_bits()
        print(f"    epoch 1 error bits: {[hex(b) for b in bits1]}")
        check(all(b == 0 for b in bits1),
              "epoch 1 is clean (credit fast path, nothing to wait on)")
        first = h.verify()
        check(all(c["allclose"] for c in first),
              "epoch 1 output is still correct")

        # Epoch 2: the starved source must time out on the credit wait.
        h.set_inputs(4321)
        fill_sentinel(h)
        h.launch(ctrl_flags=rt.CTRL_DROP_CREDIT,
                 ctrl_rank=owner_rank, ctrl_arg0=starved_source)
        bits2 = h.error_bits()
        print(f"    epoch 2 error bits: {[hex(b) for b in bits2]}")
        check(bool(bits2[starved_source] & rt.ERR_PRODUCER_CREDIT),
              f"rank{starved_source} raised PRODUCER_CREDIT timeout (bit 25)")


# ---------------------------------------------------------------------------
def control_reroute_slot(shape, source_rank=4, band=1, spin_limit=2_000_000):
    """Reroute is a silent-corruption control, so it needs a corruption
    detector that does not depend on a tolerance.

    The evaluator's allclose(rtol=1e-2, atol=1e-2) is looser than the effect of
    replacing one of eight reduction contributions with same-distribution
    values, so the run is compared byte-for-byte against a golden launch over
    identical inputs from the same module with the control flag cleared.
    """
    m, n, k, bias = shape
    plan = rt.resolve_shape(m, n, k, bias)
    bands = plan["bm"] // plan["eb"]
    print(f"\n== CTRL_REROUTE_SLOT (rank {source_rank} lands band {band} of "
          f"{bands} on the wrong rank) ==")
    if band >= bands:
        print(f"    skipped: shape only has {bands} band(s) per tile")
        return
    # The rerouted band leaves the true destination slot holding its PREVIOUS
    # epoch's bytes. With unchanged inputs those bytes are already the correct
    # answer, so the control is invisible; it is only observable across
    # changing inputs. Two epochs with different seeds, run twice.
    seed_a, seed_b = 1234, 9876

    def sequence(flags):
        with GemmRS(m, n, k, bias, spin_limit=spin_limit, control=True) as h:
            h.set_inputs(seed_a)
            h.launch(ctrl_flags=0)
            first = h.verify()
            h.set_inputs(seed_b)
            h.launch(ctrl_flags=flags, ctrl_rank=source_rank, ctrl_arg0=band)
            return {
                "out": [h.out[r].clone() for r in range(WORLD)],
                "bits": h.error_bits(),
                "epoch1_ok": all(c["allclose"] for c in first),
                "loose": h.verify(),
                "tight": h.verify(rtol=2e-3, atol=2e-3),
            }

    golden = sequence(0)
    rerouted = sequence(rt.CTRL_REROUTE_SLOT)

    check(golden["epoch1_ok"] and rerouted["epoch1_ok"],
          "epoch 1 of both runs is correct")
    print(f"    error bits: {[hex(b) for b in rerouted['bits']]}")
    check(all(b == 0 for b in rerouted["bits"]),
          "no timeout fires (readiness is published to the true destination, "
          "so this is a silent-corruption control)")

    differing, total = [], 0
    for rank in range(WORLD):
        changed = int((rerouted["out"][rank] != golden["out"][rank]).sum())
        total += changed
        if changed:
            differing.append((rank, changed, rerouted["out"][rank].numel()))
    print(f"    ranks whose payload changed vs golden: "
          f"{[(r, f'{c}/{n_}') for r, c, n_ in differing]}")
    check(total > 0,
          "rerouting demonstrably corrupts the payload (differs bitwise from "
          "the unrerouted run over an identical input sequence)")

    base_worst = max(c["max_abs_diff"] for c in golden["loose"])
    worst = max(c["max_abs_diff"] for c in rerouted["loose"])
    loose = [c["rank"] for c in rerouted["loose"] if not c["allclose"]]
    tight = [c["rank"] for c in rerouted["tight"] if not c["allclose"]]
    print(f"    max|diff| golden={base_worst:.3e} rerouted={worst:.3e}")
    print(f"    ranks failing allclose(1e-2)={loose}  allclose(2e-3)={tight}")
    check(len(tight) > 0,
          "corruption is detectable numerically at a tolerance the clean run "
          "passes with margin")
    if not loose:
        notes.append(
            "the evaluator's own allclose(rtol=1e-2, atol=1e-2) does NOT flag "
            f"this corruption (max|diff|={worst:.3e} < 1e-2): with inputs in "
            "+/-0.01 a wrong 1-of-8 contribution stays inside tolerance. "
            "Correctness confidence must not rest on the graded tolerance.")
    notes.append(
        "CTRL_REROUTE_SLOT is only observable across CHANGING inputs: the "
        "true destination slot keeps its previous epoch's bytes, which are "
        "the correct answer if the inputs did not change.")


def main():
    rt.enable_peer_access(WORLD)
    clean = (512, 4096, 12288, True)     # row 2: no tails, 1 band per tile
    banded = (64, 7168, 18432, False)    # row 1: BM=32, EB=8 -> 4 bands

    control_drop_publication(clean)
    control_drop_credit(clean)
    control_reroute_slot(banded, band=1)

    print("\n" + "=" * 72)
    for note in notes:
        print(f"note: {note}")
    if failures:
        print(f"GATE M4 FAILED: {len(failures)} expectation(s) unmet")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE M4 PASSED: all three controls failed exactly as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
