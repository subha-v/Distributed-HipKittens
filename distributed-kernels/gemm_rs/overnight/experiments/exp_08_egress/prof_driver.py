"""One (module, shape) counter-collection cell for exp_08, own process.

Deliberately NOT a timing harness: under rocprofv3 counter collection the
dispatches are instrumented and possibly serialized, so any wall/device time
measured here is meaningless. What IS meaningful is the per-dispatch transaction
counts, which are clock- and schedule-independent.

Two things this must prove before its numbers may be used:
  1. `errors` is empty -- if profiler serialization starves the credit/ready
     waits, the bounded spins time out, the sticky error bit is set, and the
     kernel returns early WITHOUT emitting, which would silently deflate every
     traffic counter.
  2. `correct` is true -- same reason, from the other side.

Usage: prof_driver.py <module> <m> <n> <k> <bias> <seed> [warm] [meas]
Prints a PROF line naming the dispatch window that the aggregator must keep.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "harness"))

import harness_lib as H  # noqa: E402
from harness_lib import rt, WORLD, GemmRS  # noqa: E402


def main():
    module_name = sys.argv[1]
    m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    bias = bool(int(sys.argv[5]))
    seed = int(sys.argv[6])
    warm = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    meas = int(sys.argv[8]) if len(sys.argv) > 8 else 4

    rt.enable_peer_access(WORLD)
    print(f"CELL {module_name} {m}x{n}x{k} bias={int(bias)} "
          f"warm={warm} meas={meas}", flush=True)
    with GemmRS(m, n, k, bias, module_name=module_name) as h:
        h.set_inputs(seed)
        for _ in range(warm):
            h.launch()
        errors = h.error_report()
        print(f"  after warmup errors: {errors or 'none'}", flush=True)
        for _ in range(meas):
            h.launch()
        errors = h.error_report()
        checks = h.verify()
        correct = all(c["allclose"] for c in checks)
        worst = max(c["max_abs_diff"] for c in checks)
        tight = all(c["allclose"] for c in h.verify(rtol=2e-3, atol=2e-3))
        # Every rank runs one dispatch per launch, so the per-agent dispatch
        # ordinals of the measured window are [warm, warm+meas).
        print(f"PROF module={module_name} shape={m}x{n}x{k} "
              f"warm={warm} meas={meas} "
              f"correct={int(correct)} tight={int(tight)} "
              f"maxdiff={worst:.3e} "
              f"errors={';'.join(errors) or 'none'} "
              f"tiles_per_cta={h.plan['gemm_tiles']}/{h.plan['num_gemm_ctas']} "
              f"eb={h.plan['eb']} nr={h.plan['num_reducer_ctas']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
