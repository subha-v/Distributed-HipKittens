"""How many of the seven xGMI egress links does a rank use AT ANY ONE TIME?

Pure arithmetic on the shipped tile map -- no GPU, no kernel, no measurement
uncertainty. Replicates, verbatim:

  gemm_rs_mi300x.cpp:213-221   the tile loop and the WGM=4 group swizzle
  gemm_rs_mi300x.cpp:281-289   row0 = tm*BM + b*EB, dest = owner_of_row(row0)
  gemm_rs_mi300x_constants.cuh owner_of_row(row, slice) = row / slice

A producer CTA runs `for (t = pid; t < tiles; t += NG)`, so the NG tiles of
round `i` are exactly {pid + i*NG}. Every rank uses the same map, so the set of
destinations that round touches is the set of egress links that rank has live.
If a round touches 2 destinations, that rank's egress is capped at 2 links no
matter how many CTAs are storing.
"""

import collections
import sys

WORLD = 8
CU = 304

# (label, M, N, K, BM, BN, NR) -- BM/BN/NR from gemm_rs_mi300x_host_abi.hpp's
# scored table as landed by exp_02 (uniform NR=32) and exp_04a (row 1 retile).
SHAPES = [
    ("1  64x7168x18432", 64, 7168, 18432, 32, 64, 32),
    ("2  512x4096x12288", 512, 4096, 12288, 64, 64, 32),
    ("3  2048x2880x2880", 2048, 2880, 2880, 128, 256, 32),
    ("4  4096x4096x4096", 4096, 4096, 4096, 256, 256, 32),
    ("5  8192x4096x14336", 8192, 4096, 14336, 256, 256, 32),
    ("6  8192x8192x29568", 8192, 8192, 29568, 256, 256, 32),
]


def decode(t, num_pid_m, num_pid_n, wgm):
    in_group = wgm * num_pid_n
    group = t // in_group
    first = group * wgm
    gsize = min(num_pid_m - first, wgm)
    tm = first + (t % in_group) % gsize
    tn = (t % in_group) // gsize
    return tm, tn


def analyse(label, m, n, k, bm, bn, nr, wgm_mode):
    ng = CU - nr
    num_pid_m = m // bm
    num_pid_n = (n + bn - 1) // bn
    tiles = num_pid_m * num_pid_n
    slice_rows = m // WORLD
    eb = min(bm, slice_rows)
    bands = bm // eb
    wgm = 4 if wgm_mode == "wgm4" else num_pid_m

    rounds = []
    seen = set()
    i = 0
    while i * ng < tiles:
        hist = collections.Counter()
        for pid in range(ng):
            t = pid + i * ng
            if t >= tiles:
                continue
            tm, tn = decode(t, num_pid_m, num_pid_n, wgm)
            assert (tm, tn) not in seen, f"{label}: tile ({tm},{tn}) visited twice"
            seen.add((tm, tn))
            for b in range(bands):
                row0 = tm * bm + b * eb
                hist[row0 // slice_rows] += 1
        rounds.append(hist)
        i += 1
    assert len(seen) == tiles, f"{label}: covered {len(seen)} of {tiles} tiles"

    # Per round: how many distinct destinations, and how skewed. The slowest
    # link is the round's critical path, so `worst share` is what bounds it.
    lines = []
    for i, hist in enumerate(rounds):
        tot = sum(hist.values())
        remote = sum(v for d, v in hist.items() if True)
        nd = len(hist)
        worst = max(hist.values()) / tot
        lines.append(f"      round {i}: links={nd}  "
                     f"worst-link share={worst * 100:4.1f}%  "
                     f"dests={dict(sorted(hist.items()))}")
    # Effective concurrent links, harmonic-style: a round of T bytes whose
    # busiest link carries share s finishes in s*T/rate, i.e. behaves like 1/s
    # links. Average that over rounds weighted by round size.
    tot_all = sum(sum(h.values()) for h in rounds)
    eff = 0.0
    for h in rounds:
        t = sum(h.values())
        eff += (t / tot_all) * (1.0 / (max(h.values()) / t))
    return lines, eff, tiles, ng, num_pid_m, num_pid_n, bands


def main():
    for mode in ("wgm4", "colmajor"):
        print("=" * 94)
        print(f"tile order = {mode}"
              + ("   (SHIPPED: constexpr WGM = 4)" if mode == "wgm4"
                 else "   (PROPOSED: WGM = num_pid_m)"))
        print("=" * 94)
        for label, m, n, k, bm, bn, nr in SHAPES:
            lines, eff, tiles, ng, npm, npn, bands = analyse(
                label, m, n, k, bm, bn, nr, mode)
            print(f"  {label}  BM={bm} BN={bn} tiles={tiles} "
                  f"({npm}x{npn}) ng={ng} bands={bands} "
                  f"rounds={len(lines)}")
            for ln in lines[:6]:
                print(ln)
            if len(lines) > 6:
                print(f"      ... {len(lines) - 6} more rounds")
            print(f"      ==> effective concurrent egress links "
                  f"(1/worst-share, round-weighted): {eff:.2f} of 8")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
