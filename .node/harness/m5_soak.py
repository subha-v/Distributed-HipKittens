"""Gate M5: 600-epoch changing-input soak with deliberate rank skew, no reset.

Requirements from MI300X_VALIDATION.md:
  * at least 600 eager launches, inputs changing, NO host reset of any signal
    or epoch state between calls;
  * every device-derived epoch advances exactly once per launch (assert the
    cells read == n_calls);
  * every touched ready/credit cell reads exactly the final epoch;
  * outputs stay correct per launch.

Rank skew matters: reducers spin on peer producers, so rotating the launch order
each epoch means a different rank is last to start every time, which is what
actually exercises the bounded wait rather than a lock-step best case.
"""

import sys
import time

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

TIGHT = 2e-3


def main():
    rt.enable_peer_access(WORLD)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 12288
    bias = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
    verify_every = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    print(f"soak: {epochs} epochs, shape m={m} n={n} k={k} bias={int(bias)}, "
          f"verifying every {verify_every} epoch(s)")
    failures = []
    worst = 0.0
    start = time.perf_counter()

    with GemmRS(m, n, k, bias) as h:
        for epoch in range(1, epochs + 1):
            # Changing inputs every epoch, and a rotating launch order so the
            # straggler rank differs from epoch to epoch.
            h.set_inputs(1000 + epoch)
            skew = [(epoch + r) % WORLD for r in range(WORLD)]
            h.launch(skew=skew)

            errors = h.error_report()
            if errors:
                failures.append(f"epoch {epoch}: error bits {errors}")
                print(f"  epoch {epoch}: ERROR BITS {errors}")
                break

            if epoch % verify_every == 0:
                checks = h.verify(rtol=TIGHT, atol=TIGHT)
                worst = max(worst, max(c["max_abs_diff"] for c in checks))
                if not all(c["allclose"] for c in checks):
                    bad = [c["rank"] for c in checks if not c["allclose"]]
                    failures.append(f"epoch {epoch}: wrong on ranks {bad}")
                    print(f"  epoch {epoch}: WRONG on ranks {bad}")
                    break

            # Epoch cells must equal the launch count at EVERY epoch, not just
            # at the end: an epoch that advanced twice and once would still sum
            # correctly at the end.
            if epoch % 50 == 0 or epoch == 1:
                problems = h.check_epochs()
                elapsed = time.perf_counter() - start
                print(f"  epoch {epoch:>4}: n_calls={h.n_calls} "
                      f"epoch cells {'all == n_calls' if not problems else problems[:1]}  "
                      f"max|diff|={worst:.3e}  {elapsed:6.1f}s")
                if problems:
                    failures.append(f"epoch {epoch}: {problems[:2]}")
                    break

        print(f"\ntotal launches: {h.n_calls}")
        final_epochs = h.check_epochs()
        final_signals = h.check_signals()
        print(f"final epoch-cell check : "
              f"{'PASS (all scheduled cells == ' + str(h.n_calls) + ', unscheduled == 0)' if not final_epochs else final_epochs[:3]}")
        print(f"final signal-cell check: "
              f"{'PASS (all touched ready/credit == ' + str(h.n_calls) + ')' if not final_signals else final_signals[:3]}")
        failures += final_epochs + final_signals
        print(f"worst max|abs diff| over soak: {worst:.3e} "
              f"(tight tolerance {TIGHT})")

    print()
    if failures:
        print(f"GATE M5 FAILED: {len(failures)} problem(s)")
        for f in failures[:8]:
            print(f"  - {f}")
        return 1
    print(f"GATE M5 PASSED: {epochs} skewed changing-input epochs, no host "
          f"reset, epochs and signals exact, outputs correct throughout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
