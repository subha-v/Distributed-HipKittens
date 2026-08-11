"""Gate M7 (part 1): per-shape timing of this port on the six graded shapes.

Order-rotated: the six shapes are measured in `rotations` different cyclic
orders in the same process, and the per-shape mean is taken across rotations,
so a shape is never systematically first (cold caches, clock ramp) or last.

Every timed run is also verified and its error bits and epoch cells checked, so
a number can never be reported for a run that silently failed.

Reports the geometric mean, because that is the competition's ranking statistic.
"""

import json
import math
import statistics
import sys

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

# Published speed-of-light table from task.yml, microseconds.
SOL = [6.46, 8.19, 23.04, 65.54, 131.07, 379.43]


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def measure(shape, iters, single_iters, reducer_ctas=None):
    m, n, k, bias, seed = shape
    with GemmRS(m, n, k, bias, num_reducer_ctas=reducer_ctas) as h:
        h.set_inputs(seed)
        h.launch()
        checks = h.verify()
        tight = h.verify(rtol=2e-3, atol=2e-3)
        pipelined = h.time_pipelined(iters=iters)
        host = h.time_host_issue()
        single = h.time_single_shot(iters=single_iters)
        epoch_problems = h.check_epochs()
        return {
            "correct": all(c["allclose"] for c in checks),
            "correct_tight": all(c["allclose"] for c in tight),
            "max_abs_diff": max(c["max_abs_diff"] for c in checks),
            "pipelined": pipelined,
            "host": host,
            "single": single,
            "epoch_problems": epoch_problems[:2],
            "plan": {key: h.plan[key] for key in
                     ("config_row", "bm", "bn", "bk", "eb", "num_reducer_ctas",
                      "gemm_tiles", "red_tiles", "lds_bytes")},
        }


def main():
    rt.enable_peer_access(WORLD)
    rotations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    single_iters = 20

    samples = {i: [] for i in range(len(SCORED))}
    device_samples = {i: [] for i in range(len(SCORED))}
    detail = {}

    for rotation in range(rotations):
        order = [(rotation + i) % len(SCORED) for i in range(len(SCORED))]
        print(f"\n===== rotation {rotation} (shape order {order}) =====")
        for index in order:
            shape = SCORED[index]
            result = measure(shape, iters, single_iters)
            detail[index] = result
            wall = result["pipelined"]["wall_us"]
            dev = result["pipelined"]["device_us_max"]
            samples[index].append(wall)
            device_samples[index].append(dev)
            status = "ok" if result["correct_tight"] else "CORRECTNESS FAIL"
            errors = result["pipelined"]["errors"]
            print(f"  m={shape[0]:<5} n={shape[1]:<5} k={shape[2]:<6} "
                  f"bias={int(shape[3])}  pipelined={wall:9.2f} us  "
                  f"device_max={dev:9.2f} us  "
                  f"single_wall={result['single']['wall_mean_us']:9.2f} us  "
                  f"skew={result['single']['launch_skew_us']:6.1f} us  "
                  f"{status}{' ERRBITS:' + str(errors) if errors else ''}")

    print("\n" + "=" * 108)
    print("GATE M7 - this port, six graded shapes, order-rotated "
          f"({rotations} rotations x {iters} pipelined iters)")
    print("=" * 108)
    header = (f"{'#':>2}{'m':>6}{'n':>6}{'k':>7}{'bias':>5}{'row':>4}"
              f"{'NR':>4}{'mean us':>10}{'stdev':>8}{'dev_max':>9}"
              f"{'hostIss':>9}{'SOL us':>9}{'x SOL':>7}{'bound':>7}{'ok':>4}")
    print(header)
    print("-" * len(header))
    means, devmeans = [], []
    all_ok = True
    for index, shape in enumerate(SCORED):
        mean = statistics.mean(samples[index])
        stdev = statistics.stdev(samples[index]) if len(samples[index]) > 1 else 0.0
        devmean = statistics.mean(device_samples[index])
        host_issue = detail[index]["host"]["host_issue_us_per_op"]
        means.append(mean)
        devmeans.append(devmean)
        ok = detail[index]["correct_tight"] and not detail[index]["epoch_problems"]
        all_ok = all_ok and ok
        # If issuing eight launches from one CPU thread costs more than the
        # device work, the measurement is reporting the host, not the kernel.
        bound = "HOST" if host_issue > 0.9 * mean else "device"
        print(f"{index+1:>2}{shape[0]:>6}{shape[1]:>6}{shape[2]:>7}"
              f"{int(shape[3]):>5}{detail[index]['plan']['config_row']:>4}"
              f"{detail[index]['plan']['num_reducer_ctas']:>4}"
              f"{mean:>10.2f}{stdev:>8.2f}{devmean:>9.2f}"
              f"{host_issue:>9.2f}{SOL[index]:>9.2f}{mean/SOL[index]:>7.2f}"
              f"{bound:>7}{'yes' if ok else 'NO':>4}")
    print("-" * len(header))
    gm = geomean(means)
    gm_dev = geomean(devmeans)
    gm_sol = geomean(SOL)
    print(f"geometric mean (pipelined wall) : {gm:9.2f} us")
    print(f"geometric mean (device max)     : {gm_dev:9.2f} us")
    print(f"geometric mean (published SOL)  : {gm_sol:9.2f} us")
    print(f"ratio to SOL                    : {gm/gm_sol:9.2f}x")
    print(f"all shapes correct              : {all_ok}")

    with open("m7_results.json", "w") as handle:
        json.dump({
            "rotations": rotations,
            "iters": iters,
            "means_us": means,
            "device_max_us": devmeans,
            "samples_us": {str(k): v for k, v in samples.items()},
            "sol_us": SOL,
            "geomean_us": gm,
            "geomean_device_us": gm_dev,
            "geomean_sol_us": gm_sol,
            "all_correct": all_ok,
            "detail": {str(k): {kk: vv for kk, vv in v.items()
                                if kk != "plan"} for k, v in detail.items()},
        }, handle, indent=2, default=str)
    print("\nwrote m7_results.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
