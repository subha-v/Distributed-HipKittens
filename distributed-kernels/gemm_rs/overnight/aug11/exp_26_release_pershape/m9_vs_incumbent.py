"""Gate M9 with a VALID golden: the incumbent rule built from the current source.

Two independent reasons the stock `harness/m9_stale_slot.py` cannot grade exp_26
as staged, both established before this driver was written and both properties
of the gate rather than of any candidate:

**1. Its frozen golden carries a superseded shape table.** The golden
`gemm_rs_mi300x_e3base` was copied before exp_14 (E4b) retiled rows 1-3. Its
dispatch is `32/64/64`, `64/64/64`, `128/256/32` where the current source and the
host-side plan say `32/64/128`, `64/128/64`, `128/192/32`. Row 1's change is
BK-only, so its tile map and signal geometry are untouched and the golden stays
bit-exact there -- which is why that case passes. Rows 2 and 3 changed BN:

    row 2   N=4096   golden cols 64 vs plan cols 32   tile map differs
    row 3   N=2880   golden cols 12 x BN 256 = 3072 B rows of a 2880-row operand

So on row 2 the golden reduces against a map the harness did not allocate for and
reports NaN plus 600/600 bitwise differences, and on row 3 it reads 192 rows past
the end of B. The 10:05Z `Memory access fault by GPU node-7` came from that read
-- it is the exact out-of-bounds access exp_14's own shape-table comment says the
retile removed, preserved in amber by the frozen module.

**2. Even repaired, "pre-E3" is the wrong comparand for this change.** exp_26
does not ask whether the kernel still matches a build from before release
grouping; it asks whether changing the group RULE moves a bit. The right golden
is `gemm_rs_mi300x_ps0`: the same source, the same shape table, the same tile
order, PERSHAPE=0. Bit-identity against it is protocol-review condition C5/C8
stated exactly.

Everything else is m9's own machinery, untouched: same poison, same detectors,
same CTRL_PUBLISH_EARLY, same pass/fail logic. `harness/m9_stale_slot.py` is not
modified.

  python3 m9_vs_incumbent.py [scale]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import m9_stale_slot as M9                                # noqa: E402

M9.GOLDEN_MODULE = "gemm_rs_mi300x_ps0"

# All six graded rows plus the generic row. Shape 5 leads and gets the most
# epochs because it is the ONLY row whose group size this change moves: 512
# tiles over 272 producers = 2 tiles per CTA, with 32 CTAs owning a 1-tile tail,
# so it is also the only row that exercises a group of exactly 2 and its
# truncation. The stock case list has not covered it since exp_14 retiled row 2
# to one tile per CTA.
M9.CASES = [
    ((8192, 4096, 14336, True), 120,
     "row 5, 512 tiles / 272 producers = 2 tiles/CTA + 1-tile tail -- "
     "THE ONLY ROW exp_26 CHANGES (rgroup 1 -> 2)"),
    ((8192, 8192, 29568, False), 50, "row 6, 4 tiles/CTA + 3-tile tail"),
    ((8192, 8192, 28672, False), 20, "generic row, up to 118 tiles/CTA"),
    ((512, 4096, 12288, True), 300, "row 2, 1 tile/CTA (C8 control)"),
    ((2048, 2880, 2880, True), 40, "row 3, 1 tile/CTA (C8 control)"),
    ((64, 7168, 18432, False), 30, "row 1, 1 tile/CTA, 24 idle CTAs (C8)"),
    ((4096, 4096, 4096, False), 30, "row 4, 1 tile/CTA, 16 idle CTAs (C8)"),
]
M9.CONTROL_EPOCHS = [30, 20, 10, 20, 10, 10, 10]

if __name__ == "__main__":
    sys.argv = sys.argv[:2]
    sys.exit(M9.main())
