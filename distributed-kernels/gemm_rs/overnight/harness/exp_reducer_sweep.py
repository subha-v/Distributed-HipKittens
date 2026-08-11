"""Tuning experiment 1: the NUM_REDUCER_CTAS split, per shape.

MI300X_VALIDATION.md lists this first among the known simplifications: the
table's reducer counts are RadeonFlow's submitted values (32/48/48/48/32/8),
carried over as an initial configuration rather than a measured optimum. The
split only changes how the fixed 304 CTAs are partitioned; the dependency
layout (EB, lrow/col counts, ready/credit indices) is independent of it, so
nothing else has to move.

Correctness is re-checked at every split, because a bad split is a protocol
stress test, not just a slower run: fewer reducers means more tiles per reducer
and longer producer-side credit waits.
"""

import json
import sys

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

SHAPES = [
    (64, 7168, 18432, False, 1234, 32),
    (512, 4096, 12288, True, 663, 48),
    (2048, 2880, 2880, True, 166, 48),
    (4096, 4096, 4096, False, 1371, 48),
    (8192, 4096, 14336, True, 7168, 32),
    (8192, 8192, 29568, False, 42, 8),
]
SPLITS = [8, 16, 24, 32, 40, 48]
TIGHT = 2e-3


def main():
    rt.enable_peer_access(WORLD)
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    results = {}
    print(f"{'shape':<22}{'red_tiles':>10}" +
          "".join(f"{f'NR={s}':>12}" for s in SPLITS) +
          f"{'best':>8}{'table':>7}{'gain':>8}")
    print("-" * (22 + 10 + 12 * len(SPLITS) + 23))
    for m, n, k, bias, seed, table_nr in SHAPES:
        row = {}
        red_tiles = rt.resolve_shape(m, n, k, bias)["red_tiles"]
        for split in SPLITS:
            try:
                with GemmRS(m, n, k, bias, num_reducer_ctas=split) as h:
                    h.set_inputs(seed)
                    h.launch()
                    ok = all(c["allclose"]
                             for c in h.verify(rtol=TIGHT, atol=TIGHT))
                    timing = h.time_pipelined(iters=iters)
                    errors = h.error_report() + timing["errors"]
                    row[split] = {"us": timing["wall_us"], "ok": ok and not errors,
                                  "errors": errors}
            except Exception as exc:  # noqa: BLE001
                row[split] = {"us": float("nan"), "ok": False,
                              "errors": [f"{type(exc).__name__}: {exc}"]}
        results[f"{m}x{n}x{k}"] = row

        valid = {s: v["us"] for s, v in row.items() if v["ok"]}
        best = min(valid, key=valid.get) if valid else None
        table_us = row.get(table_nr, {}).get("us", float("nan"))
        gain = (table_us / valid[best] - 1) * 100 if best else float("nan")
        cells = "".join(
            f"{row[s]['us']:>11.1f}{'*' if not row[s]['ok'] else ' '}"
            for s in SPLITS)
        print(f"{f'{m}x{n}x{k}':<22}{red_tiles:>10}{cells}"
              f"{best if best else '-':>8}{table_nr:>7}{gain:>7.1f}%",
              flush=True)

    print("-" * 60)
    print("'*' marks a split that failed correctness or raised an error bit.")
    print("'table' is the value currently in the shape table; 'gain' is how "
          "much the best split beats it.")
    with open("reducer_sweep.json", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print("wrote reducer_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
