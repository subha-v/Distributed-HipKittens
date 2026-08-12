"""Print the resolved plan per graded shape, with the wave/round arithmetic.

The point of this file is to read the split back out of the compiled runtime
rather than out of the header, so the table, the binary and the geometry are
confirmed together.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

from harness_lib import rt                                      # noqa: E402

SHAPES = [
    (64, 7168, 18432, False), (512, 4096, 12288, True),
    (2048, 2880, 2880, True), (4096, 4096, 4096, False),
    (8192, 4096, 14336, True), (8192, 8192, 29568, False),
]
EXPECT_NR = [56, 32, 32, 32, 32, 48]


def main():
    print(f"{'shape':<22}{'row':>4}{'tile':>12}{'NR':>4}{'NG':>5}"
          f"{'tiles':>8}{'waves':>7}{'red':>6}{'rounds':>8}{'WGM':>5}{'ok':>4}")
    bad = 0
    for (m, n, k, bias), want in zip(SHAPES, EXPECT_NR):
        p = rt.resolve_shape(m, n, k, bias)
        ng, nr = int(p["num_gemm_ctas"]), int(p["num_reducer_ctas"])
        tiles, red = int(p["gemm_tiles"]), int(p["red_tiles"])
        tile = "{}/{}/{}".format(p["bm"], p["bn"], p["bk"])
        wgm = 4 if tiles <= ng else m // int(p["bm"])
        ok = nr == want
        bad += not ok
        print(f"{f'{m}x{n}x{k}':<22}{p['config_row']:>4}{tile:>12}{nr:>4}"
              f"{ng:>5}{tiles:>8}{-(-tiles // ng):>7}{red:>6}"
              f"{-(-red // nr):>8}{wgm:>5}{'yes' if ok else 'NO':>4}")
    gen = rt.resolve_shape(2048, 4096, 7168, False)
    print(f"generic row (unchanged, must be 24): NR={gen['num_reducer_ctas']} "
          f"row={gen['config_row']}")
    print("PLAN CHECK PASSED" if bad == 0 and gen["num_reducer_ctas"] == 24
          else "PLAN CHECK FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
