"""E4b: exhaustive enumeration of the legal (BM, BN, BK) space per scored shape.

Pure host arithmetic, no GPU. Every constraint below was read out of the source
rather than taken from the brief; the source line is named next to each one so a
future reader can re-verify without re-deriving:

  gemm_rs_mi300x.cpp:204   BM % 32 == 0, BN % 64 == 0   (WARPS_M=2, WARPS_N=4)
  gemm_rs_mi300x.cpp:208   LDS = 2*(BM+BN)*BK*2 <= 65536
  gemm_rs_mi300x.cpp:215   STAGE_BYTES_MAX = 32*BN*2 <= LDS_BYTES
  gemm_rs_mi300x.cpp:257   KS = BK/2, KS % 16 == 0      => BK % 32 == 0
  hk_adapter.cuh:120       g2s big_calls == 1           => BM*BK, BN*BK <= 65536
  host_abi.hpp:169         BM must divide M
  host_abi.hpp:174         lrow_count <= 32, col_count <= 128
  constants.cuh:24         LROW_MAX = 32, COL_MAX = 128

THE COST MODELS, and why there are three of them.

E4 established that a strided persistent loop costs ROUNDS, not CTAs. Applied to
the tile dimension the same statement is: the producer critical path is
`waves` tile-bodies deep, where waves = ceil(tiles / NG). But unlike NR, moving
BM/BN/BK also changes what ONE tile body costs, so `waves` alone is not a cost
-- it has to be multiplied by a per-tile cost, and which per-tile cost is right
depends on what the mainloop is actually bound by. Hence three columns:

  wave_cost = waves * BM*BN            MFMA-bound: a tile body is
                                       k_iters * BM*BN*BK/1024 cycles, and
                                       k_iters*BK ~ K_local, so the whole
                                       producer path is waves*BM*BN*K_local.
                                       Floor is M*N/NG (perfect fill), so
                                       wave_cost = (M*N/NG) / avg_fill: this is
                                       exactly the brief's fill criterion,
                                       re-derived as a time rather than as an
                                       occupancy.

  iter_cost = waves * k_iters          OVERHEAD-bound: every k-iteration pays a
                                       full __syncthreads(), a vmcnt(0), an
                                       lgkmcnt(0) and the LDS round trip
                                       whatever the tile size. Shape 6 measures
                                       ~4700 cycles/iteration against ~2050 of
                                       MFMA, so this term is the LARGER half
                                       today and the brief's model omits it.

  traffic   = (1/BM + 1/BN)            BANDWIDTH-bound: global bytes are
                                       M*N*K_local*2*(1/BM + 1/BN) before L2
                                       reuse, so shrinking a tile to buy fill
                                       buys it with DRAM traffic. This is the
                                       term that kills the "just use BN=64"
                                       moves that look free on wave_cost.

A candidate is only pre-registered as a predicted win if it improves at least
one column materially and does not badly regress the others.
"""

import math
from math import gcd

CU = 304
LROW_MAX, COL_MAX = 32, 128

# (M, N, K_global, has_bias, landed NR, landed BM/BN/BK)
SHAPES = [
    (64, 7168, 18432, False, 56, (32, 64, 64)),
    (512, 4096, 12288, True, 32, (64, 64, 64)),
    (2048, 2880, 2880, True, 32, (128, 256, 32)),
    (4096, 4096, 4096, False, 32, (256, 256, 32)),
    (8192, 4096, 14336, True, 32, (256, 256, 32)),
    (8192, 8192, 29568, False, 48, (256, 256, 32)),
]


def geometry(m, n, k_global, nr, bm, bn, bk):
    """Everything resolve_shape would derive, plus the three cost columns.

    Returns None with a reason when a gate rejects the combination.
    """
    k_local = k_global // 8
    slice_rows = m // 8
    if bm % 32 or bn % 64 or bk % 32:
        return None, "granularity"
    if m % bm:
        return None, "BM does not divide M"
    lds = 2 * (bm + bn) * bk * 2
    if lds > 65536:
        return None, f"LDS {lds}"
    if 32 * bn * 2 > lds:
        return None, "stage window > LDS"
    if bm * bk > 65536 or bn * bk > 65536:
        return None, "g2s big_calls > 1"
    eb = gcd(bm, slice_rows)
    if eb <= 0 or slice_rows % eb or bm % eb:
        return None, "emit band"
    lrows = slice_rows // eb
    cols = (n + bn - 1) // bn
    if lrows > LROW_MAX or cols > COL_MAX:
        return None, f"signal geometry lrows={lrows} cols={cols}"
    ng = CU - nr
    tiles = (m // bm) * cols
    waves = (tiles + ng - 1) // ng
    red_tiles = lrows * cols
    rounds = (red_tiles + nr - 1) // nr
    k_iters = (k_local + bk - 1) // bk
    k_tail = (k_local % bk) != 0
    accum_vgpr = bm * bn // 512
    win_rows = min(eb, 32)
    return {
        "bm": bm, "bn": bn, "bk": bk, "lds": lds, "eb": eb, "lrows": lrows,
        "cols": cols, "tiles": tiles, "waves": waves, "red_tiles": red_tiles,
        "rounds": rounds, "k_iters": k_iters, "k_tail": k_tail, "ng": ng,
        "last_fill": (tiles - (waves - 1) * ng) / ng,
        "avg_fill": tiles / (waves * ng),
        "n_pad": cols * bn / n,
        "accum_vgpr": accum_vgpr,
        "scan_amp": bm / win_rows,          # staging rescans per useful row
        "wave_cost": waves * bm * bn,
        "iter_cost": waves * k_iters,
        "traffic": (1.0 / bm + 1.0 / bn),
        "oob_b_rows": cols * bn - n,        # B rows read past the end of w
    }, None


def sweep(idx):
    m, n, k_global, bias, nr, landed = SHAPES[idx]
    base, err = geometry(m, n, k_global, nr, *landed)
    assert err is None, err
    rows = []
    for bm in range(32, min(m, 1024) + 1, 32):
        for bn in range(64, 1024 + 1, 64):
            if bm * bn > 65536:                 # accumulator VGPR ceiling
                continue
            for bk in (32, 64, 96, 128, 160, 192, 256):
                g, err = geometry(m, n, k_global, nr, bm, bn, bk)
                if g is None:
                    continue
                rows.append(g)
    for g in rows:
        g["rel_wave"] = g["wave_cost"] / base["wave_cost"]
        g["rel_iter"] = g["iter_cost"] / base["iter_cost"]
        g["rel_traf"] = g["traffic"] / base["traffic"]
        # A single scalar only to order the printout: geometric blend of the two
        # time-like columns. Deliberately ignores traffic, which is reported
        # separately so it cannot be hidden inside a composite.
        g["score"] = math.sqrt(g["rel_wave"] * g["rel_iter"])
    return base, sorted(rows, key=lambda g: g["score"])


HEAD = (f"{'BM/BN/BK':>13}{'tiles':>7}{'wav':>4}{'lastf':>7}{'avgf':>7}"
        f"{'kit':>5}{'tl':>3}{'cols':>5}{'red':>5}{'rnd':>4}{'LDS':>7}"
        f"{'acc':>5}{'pad':>6}{'w*x':>7}{'w*ki':>6}{'traf':>7}{'score':>7}")


def show(g, mark=""):
    print(f"{g['bm']:>5}/{g['bn']}/{g['bk']:<4}"
          f"{g['tiles']:>7}{g['waves']:>4}{g['last_fill'] * 100:>6.0f}%"
          f"{g['avg_fill'] * 100:>6.0f}%{g['k_iters']:>5}"
          f"{int(g['k_tail']):>3}{g['cols']:>5}{g['red_tiles']:>5}"
          f"{g['rounds']:>4}{g['lds'] // 1024:>6}K{g['accum_vgpr']:>5}"
          f"{(g['n_pad'] - 1) * 100:>5.1f}%{g['rel_wave']:>7.2f}"
          f"{g['rel_iter']:>6.2f}{g['rel_traf']:>7.2f}{g['score']:>7.3f} {mark}")


def main():
    for idx in range(6):
        m, n, k_global, bias, nr, landed = SHAPES[idx]
        base, rows = sweep(idx)
        print(f"\n{'=' * 118}")
        print(f"shape {idx + 1}: {m}x{n}x{k_global}  K_local={k_global // 8} "
              f"slice={m // 8}  NR={nr} NG={base['ng']}  landed "
              f"{landed[0]}/{landed[1]}/{landed[2]}  legal points={len(rows)}")
        print(f"{'=' * 118}")
        print(HEAD)
        for g in rows[:12]:
            islanded = (g["bm"], g["bn"], g["bk"]) == landed
            show(g, "<== LANDED" if islanded else "")
        if not any((g["bm"], g["bn"], g["bk"]) == landed for g in rows[:12]):
            print("  ... landed row:")
            show(base, "<== LANDED")


if __name__ == "__main__":
    main()
