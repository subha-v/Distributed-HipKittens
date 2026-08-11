"""One (module, shape) timing cell, run in its own process.

A GPU memory fault aborts the process, so each ablation cell is isolated: the
driver can then report exactly which arm faulted instead of losing the sweep.
"""

import sys

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS


def main():
    module_name = sys.argv[1]
    m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    bias = bool(int(sys.argv[5]))
    iters = int(sys.argv[6])
    seed = int(sys.argv[7])

    rt.enable_peer_access(WORLD)
    print(f"CELL {module_name} {m}x{n}x{k} bias={int(bias)}", flush=True)
    with GemmRS(m, n, k, bias, module_name=module_name) as h:
        h.set_inputs(seed)
        print("  launching once", flush=True)
        h.launch()
        errors = h.error_report()
        print(f"  first launch errors: {errors or 'none'}", flush=True)
        checks = h.verify()
        correct = all(c["allclose"] for c in checks)
        print(f"  correct={correct} "
              f"max|diff|={max(c['max_abs_diff'] for c in checks):.3e}",
              flush=True)
        print("  timing", flush=True)
        timing = h.time_pipelined(iters=iters)
        print(f"RESULT wall_us={timing['wall_us']:.3f} "
              f"device_us={timing['device_us_max']:.3f} "
              f"correct={int(correct)} "
              f"errors={';'.join(timing['errors']) or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
