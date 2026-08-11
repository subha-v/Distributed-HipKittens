"""Gate M3: world-8 correctness against the evaluator's oracle.

Usage:
  python3 m3_correctness.py one 512 4096 12288 1     # single shape, verbose
  python3 m3_correctness.py scored                   # the six benchmark shapes
  python3 m3_correctness.py tests                    # the eleven official tests
  python3 m3_correctness.py all
"""

import sys
import traceback

import torch

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

# Graded tolerance is 1e-2; this is the tighter regression tolerance that a
# healthy run clears by an order of magnitude while protocol corruption does not.
TIGHT = 2e-3

# The six graded benchmark shapes.
SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

# The eleven official correctness cases from task.yml.
OFFICIAL = [
    (64, 2880, 2880, True, 2035),
    (64, 3584, 14336, True, 13),
    (512, 3584, 14336, True, 4297),
    (512, 4608, 36864, False, 1597),
    (2048, 4096, 7168, False, 716),
    (2048, 8192, 30720, False, 20201),
    (4096, 2880, 2880, True, 136),
    (4096, 8192, 2048, True, 138),
    (8192, 3584, 14336, True, 748),
    (8192, 4608, 36864, True, 4422),
    (8192, 8192, 28672, False, 1536),
]


def run_shape(m, n, k, has_bias, seed, epochs=3, verbose=False,
              spin_limit=2_000_000):
    """Launch `epochs` times with changing inputs; verify every epoch.

    More than one epoch matters: epoch 1 takes the credit-wait fast path, so
    only epoch 2 onward exercises slot reuse against a real retirement credit.
    """
    plan = rt.resolve_shape(m, n, k, has_bias)
    label = f"m={m} n={n} k={k} bias={int(has_bias)}"
    if verbose:
        print(f"\n=== {label} ===")
        for key in ("config_row", "bm", "bn", "bk", "eb", "lrow_count",
                    "col_count", "gemm_tiles", "red_tiles", "num_gemm_ctas",
                    "num_reducer_ctas", "even_k", "even_n", "packet_fast_path",
                    "lds_bytes", "signal_words_total", "k_local"):
            print(f"    {key:>20} = {plan[key]}")

    result = {"label": label, "plan": plan, "epochs": [], "ok": True,
              "error": None}
    harness = None
    try:
        harness = GemmRS(m, n, k, has_bias, spin_limit=spin_limit)
        for epoch in range(1, epochs + 1):
            harness.set_inputs(seed + 1000 * (epoch - 1))
            harness.launch()
            errors = harness.error_report()
            checks = harness.verify()
            worst = max(c["max_abs_diff"] for c in checks)
            allok = all(c["allclose"] for c in checks)
            bad_ranks = [c["rank"] for c in checks if not c["allclose"]]
            # The graded tolerance has little detection power at this input
            # scale (Gate M4's reroute control corrupts a whole source
            # contribution and still lands under 1e-2), so a tighter tolerance
            # is recorded alongside it as the real regression signal.
            tight = all(c["allclose"]
                        for c in harness.verify(rtol=TIGHT, atol=TIGHT))
            epoch_info = {
                "epoch": epoch,
                "error_bits": errors,
                "allclose": allok,
                "tight_allclose": tight,
                "max_abs_diff": worst,
                "bad_ranks": bad_ranks,
                "checks": checks,
            }
            result["epochs"].append(epoch_info)
            if errors or not allok:
                result["ok"] = False
            if not tight:
                result["ok"] = False
            if verbose:
                print(f"    epoch {epoch}: allclose={allok} "
                      f"tight({TIGHT})={tight} "
                      f"max|diff|={worst:.3e} errbits={errors or 'none'}")
                if not allok:
                    for c in checks:
                        if not c["allclose"]:
                            print(f"      rank{c['rank']}: "
                                  f"{c['mismatches']}/{c['numel']} mismatched, "
                                  f"max|diff|={c['max_abs_diff']:.3e}, "
                                  f"ref absmax={c['ref_absmax']:.3e}, "
                                  f"got absmax={c['got_absmax']:.3e}")

        epoch_problems = harness.check_epochs()
        signal_problems = harness.check_signals()
        result["epoch_problems"] = epoch_problems
        result["signal_problems"] = signal_problems
        if epoch_problems or signal_problems:
            result["ok"] = False
        if verbose:
            print(f"    epoch cells: "
                  f"{'all == ' + str(harness.n_calls) if not epoch_problems else epoch_problems[:3]}")
            print(f"    signal cells: "
                  f"{'all == ' + str(harness.n_calls) if not signal_problems else signal_problems[:3]}")
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
    finally:
        if harness is not None:
            harness.close()
    return result


def summarize(results):
    print("\n" + "=" * 96)
    print(f"{'shape':<34}{'row':>4}{'ep':>4}{'allclose':>10}{'tight':>8}"
          f"{'max|diff|':>12}{'errbits':>9}{'verdict':>9}")
    print("-" * 96)
    npass = 0
    for r in results:
        row = r["plan"]["config_row"] if r["plan"] else "?"
        if r["error"]:
            print(f"{r['label']:<34}{row:>4}{'-':>4}{'-':>10}{'-':>8}{'-':>12}"
                  f"{'-':>9}{'ERROR':>9}")
            print(f"    {r['error']}")
            continue
        allok = all(e["allclose"] for e in r["epochs"])
        tight = all(e["tight_allclose"] for e in r["epochs"])
        worst = max(e["max_abs_diff"] for e in r["epochs"])
        errbits = any(e["error_bits"] for e in r["epochs"])
        verdict = "PASS" if r["ok"] else "FAIL"
        npass += r["ok"]
        print(f"{r['label']:<34}{row:>4}{len(r['epochs']):>4}{str(allok):>10}"
              f"{str(tight):>8}{worst:>12.3e}{str(errbits):>9}{verdict:>9}")
        for problem in r.get("epoch_problems", [])[:2]:
            print(f"    epoch: {problem}")
        for problem in r.get("signal_problems", [])[:2]:
            print(f"    signal: {problem}")
    print("-" * 96)
    print(f"{npass}/{len(results)} shapes PASSED")
    return npass == len(results)


def main():
    rt.enable_peer_access(WORLD)
    args = sys.argv[1:] or ["scored"]
    mode = args[0]

    if mode == "one":
        m, n, k, bias = int(args[1]), int(args[2]), int(args[3]), bool(int(args[4]))
        epochs = int(args[5]) if len(args) > 5 else 3
        results = [run_shape(m, n, k, bias, 1234, epochs=epochs, verbose=True)]
    elif mode == "scored":
        results = [run_shape(*s, verbose=True) for s in SCORED]
    elif mode == "tests":
        results = [run_shape(*s, verbose=True) for s in OFFICIAL]
    elif mode == "all":
        results = [run_shape(*s, verbose=True) for s in SCORED + OFFICIAL]
    else:
        print(__doc__)
        return 2

    return 0 if summarize(results) else 1


if __name__ == "__main__":
    sys.exit(main())
