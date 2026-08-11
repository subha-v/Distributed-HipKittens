"""Experiment: does the payload heap have to be fine-grained (uncached)?

The kernel's release is a directional L2 writeback (buffer_wbl2 sc0 sc1) and its
acquire is a pure invalidate (buffer_inv), which is exactly the handshake needed
to publish a CACHED allocation to a peer. If that is sufficient in practice,
coarse-grained payload memory is both correct and much faster, because
fine-grained memory is uncached -- which penalizes even the reducer's local
reads of all eight slots.

Signals stay fine-grained in every arm except the last: they are relaxed
system-scope atomics with no surrounding fence.

Correctness is checked in every arm, over multiple epochs with changing inputs,
because a coherence shortcut that is merely usually-right is worse than useless.
"""

import sys

import harness_lib as H
from harness_lib import rt, WORLD, GemmRS

SHAPES = [
    (64, 7168, 18432, False, 1234),
    (512, 4096, 12288, True, 663),
    (2048, 2880, 2880, True, 166),
    (4096, 4096, 4096, False, 1371),
    (8192, 4096, 14336, True, 7168),
    (8192, 8192, 29568, False, 42),
]

ARMS = [
    ("payload=fine  sig=fine", dict(payload_fine=True, signal_fine=True)),
    ("payload=coarse sig=fine", dict(payload_fine=False, signal_fine=True)),
    ("payload=coarse sig=coarse", dict(payload_fine=False, signal_fine=False)),
]

EPOCHS = 4
TIGHT = 2e-3


def run(shape, arm_kwargs, iters):
    m, n, k, bias, seed = shape
    with GemmRS(m, n, k, bias, **arm_kwargs) as h:
        # Correctness first, with changing inputs so slot reuse is exercised
        # and any stale-cache read would show up as a wrong answer.
        worst, ok, tight_ok = 0.0, True, True
        for epoch in range(EPOCHS):
            h.set_inputs(seed + 977 * epoch)
            h.launch()
            checks = h.verify()
            ok = ok and all(c["allclose"] for c in checks)
            tight_ok = tight_ok and all(
                c["allclose"] for c in h.verify(rtol=TIGHT, atol=TIGHT))
            worst = max(worst, max(c["max_abs_diff"] for c in checks))
        errors = h.error_report()
        timing = h.time_pipelined(iters=iters)
        return {
            "ok": ok, "tight_ok": tight_ok, "max_abs_diff": worst,
            "us": timing["wall_us"], "device_us": timing["device_us_max"],
            "errors": errors + timing["errors"],
        }


def main():
    rt.enable_peer_access(WORLD)
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    results = {}
    for label, kwargs in ARMS:
        print(f"\n===== {label} =====")
        for shape in SHAPES:
            try:
                out = run(shape, kwargs, iters)
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "tight_ok": False, "max_abs_diff": float("nan"),
                       "us": float("nan"), "device_us": float("nan"),
                       "errors": [f"{type(exc).__name__}: {exc}"]}
            results[(label, shape[:4])] = out
            flag = "ok" if out["tight_ok"] else ("LOOSE-ONLY" if out["ok"] else "WRONG")
            print(f"  m={shape[0]:<5} n={shape[1]:<5} k={shape[2]:<6} "
                  f"{out['us']:9.2f} us  max|diff|={out['max_abs_diff']:.3e}  "
                  f"{flag}"
                  f"{'  ERR:' + str(out['errors']) if out['errors'] else ''}")

    print("\n" + "=" * 104)
    header = (f"{'shape':<26}" +
              "".join(f"{label.split()[0].split('=')[1][:6]:>11}" +
                      f"{'corr':>6}" for label, _ in ARMS) + f"{'speedup':>9}")
    print(f"{'shape':<26}" + "".join(f"{label:>28}" for label, _ in ARMS))
    print("-" * 104)
    for shape in SHAPES:
        row = f"{f'{shape[0]}x{shape[1]}x{shape[2]}':<26}"
        for label, _ in ARMS:
            out = results[(label, shape[:4])]
            mark = "ok" if out["tight_ok"] else ("~" if out["ok"] else "WRONG")
            row += f"{out['us']:>20.2f} {mark:<7}"
        print(row)
    print("-" * 104)
    print("\nspeedup of coarse payload over fine payload:")
    for shape in SHAPES:
        fine = results[("payload=fine  sig=fine", shape[:4])]["us"]
        coarse = results[("payload=coarse sig=fine", shape[:4])]["us"]
        if fine == fine and coarse == coarse and coarse > 0:
            print(f"  {shape[0]}x{shape[1]}x{shape[2]}: {fine/coarse:.2f}x "
                  f"({fine:.1f} -> {coarse:.1f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
