"""Gate M8: graph mode, valid only if device-derived epochs advance per replay.

The failure mode this guards against is RadeonFlow's: a host-supplied
`signal_val` captured into the graph replays the same epoch forever, so the
readiness comparison passes against stale signals. This port derives every epoch
from a device cell, so the property is structural -- but Gate M8 requires it be
proven with a capture whose inputs change between replays, which is what this
does.

Graph mode is also the only accurate measurement for the smaller shapes: the
eager path costs ~11 us of host time per launch, i.e. ~90 us per operation.
"""

import sys

import torch

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

SCORED = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]
SOL = [6.46, 8.19, 23.04, 65.54, 131.07, 379.43]

failures = []


def check(condition, message):
    print(f"    {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        failures.append(message)
    return condition


def epoch_correctness_under_replay(shape, reps=4, replays=5):
    """Inputs change between replays; every replay must advance every epoch."""
    m, n, k, bias, seed = shape
    print(f"\n== epoch advance + correctness under replay "
          f"(m={m} n={n} k={k}) ==")
    with GemmRS(m, n, k, bias) as h:
        h.set_inputs(seed)
        h.launch()  # one eager launch first, so slots hold live payload
        h.build_graphs(reps=reps)
        baseline = h.n_calls

        for replay_index in range(replays):
            # Change the inputs IN PLACE between replays: the graph captured
            # the pointers, so a stale-epoch bug would keep returning the old
            # answer against signals that never advanced.
            fresh = [H.generate_input(r, WORLD, m, n, k, bias,
                                      seed + 4242 * (replay_index + 1))
                     for r in range(WORLD)]
            for rank in range(WORLD):
                h.inputs[rank][0].copy_(fresh[rank][0])
                h.inputs[rank][1].copy_(fresh[rank][1])
                if h.inputs[rank][2] is not None:
                    h.inputs[rank][2].copy_(fresh[rank][2])
            torch.cuda.synchronize()

            h.replay_graphs(times=1)
            expected = baseline + reps * (replay_index + 1)
            problems = h.check_epochs(expected=expected)
            errors = h.error_report()
            checks = h.verify()
            tight = h.verify(rtol=2e-3, atol=2e-3)
            print(f"    replay {replay_index}: epochs=={expected}? "
                  f"{'yes' if not problems else problems[:1]}  "
                  f"errbits={errors or 'none'}  "
                  f"allclose={all(c['allclose'] for c in checks)}  "
                  f"max|diff|={max(c['max_abs_diff'] for c in checks):.3e}")
            check(not problems,
                  f"replay {replay_index}: every scheduled epoch cell == "
                  f"{expected}")
            check(not errors, f"replay {replay_index}: no error bits")
            check(all(c["allclose"] for c in tight),
                  f"replay {replay_index}: output correct for the NEW inputs "
                  f"(not a stale re-run)")

        signal_problems = h.check_signals()
        check(not signal_problems,
              f"every touched ready/credit cell == {h.n_calls}")


def graph_timing():
    print("\n" + "=" * 100)
    print("graph-mode timing (host launch cost amortized by the capture)")
    print("=" * 100)
    header = (f"{'#':>2}{'shape':>22}{'eager us':>10}{'graph us':>10}"
              f"{'speedup':>9}{'SOL us':>9}{'x SOL':>7}{'ok':>4}")
    print(header)
    print("-" * len(header))
    graph_means = []
    for index, shape in enumerate(SCORED):
        m, n, k, bias, seed = shape
        with GemmRS(m, n, k, bias) as h:
            h.set_inputs(seed)
            h.launch()
            eager = h.time_pipelined(iters=50)["wall_us"]
            graph = h.time_graph(reps=20, replays=20)
            correct = all(c["allclose"]
                          for c in h.verify(rtol=2e-3, atol=2e-3))
            errors = h.error_report()
            graph_means.append(graph["wall_us"])
            print(f"{index+1:>2}{f'{m}x{n}x{k}':>22}{eager:>10.2f}"
                  f"{graph['wall_us']:>10.2f}"
                  f"{eager/graph['wall_us']:>9.2f}{SOL[index]:>9.2f}"
                  f"{graph['wall_us']/SOL[index]:>7.2f}"
                  f"{'yes' if correct and not errors else 'NO':>4}")
            if errors:
                print(f"      error bits: {errors}")
    print("-" * len(header))
    import math
    gm = math.exp(sum(math.log(v) for v in graph_means) / len(graph_means))
    gm_sol = math.exp(sum(math.log(v) for v in SOL) / len(SOL))
    print(f"geometric mean (graph mode) : {gm:9.2f} us")
    print(f"geometric mean (SOL)        : {gm_sol:9.2f} us")
    print(f"ratio to SOL                : {gm/gm_sol:9.2f}x")
    return graph_means


def main():
    rt.enable_peer_access(WORLD)
    epoch_correctness_under_replay(SCORED[1])
    epoch_correctness_under_replay(SCORED[3])
    graph_timing()

    print()
    if failures:
        print(f"GATE M8 FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE M8 PASSED: epochs advance on every replay and outputs track "
          "changing inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
