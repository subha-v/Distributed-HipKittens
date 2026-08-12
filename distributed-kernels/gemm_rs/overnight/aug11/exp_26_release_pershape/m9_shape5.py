"""Gate M9, re-aimed at the ONE shape exp_26 actually changes.

`harness/m9_stale_slot.py`'s case list was written for E3, when 512x4096x12288
was the 2-tiles-per-CTA row. exp_14 retiled that row to 64/128 -- 256 tiles over
272 producers, **1 tile per CTA** -- so as of today the stock case list contains
no shape whose group size exp_26 moves:

    512x4096x12288   1 tile /CTA   rgroup 1 -> 1   unchanged
    8192x8192x29568  4 tiles/CTA   rgroup 4 -> 4   unchanged
    8192x8192x28672  118  /CTA     rgroup 4 -> 4   unchanged
    64x7168x18432    1 tile /CTA   rgroup 1 -> 1   unchanged
    2048x2880x2880   1 tile /CTA   rgroup 1 -> 1   unchanged
    4096x4096x4096   1 tile /CTA   rgroup 1 -> 1   unchanged

A gate that passes on six shapes none of which exercise the change is not
evidence about the change. **8192x4096x14336 is the whole of exp_26**: 512 tiles
over 272 producers = 2 tiles per CTA with a 1-tile tail on 32 of them, so it is
also the only shape that exercises a group of exactly 2 AND its truncated tail.

This driver reuses m9's machinery verbatim -- same poison, same bitwise golden,
same detectors, same pass/fail logic -- and only replaces the case list, so the
candidate and the CTRL_PUBLISH_EARLY control are measured by exactly the same
instrument as the stock gate. harness/m9_stale_slot.py is not modified.

  python3 m9_shape5.py [scale]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "harness"))

import m9_stale_slot as M9                                # noqa: E402

# (shape, epochs, why). 8192x4096x14336 is a ~660 us operation, so 120 epochs is
# ~80 s of device time per arm -- cheap enough to be generous with. The two
# neighbours are kept as bit-identity controls: whatever the poisoned heap says
# about shape 5 has to be said against shapes whose group size did not move.
M9.CASES = [
    ((8192, 4096, 14336, True), 120,
     "row 5, 512 tiles / 272 producers = 2 tiles/CTA with a 1-tile tail -- "
     "THE ONLY SHAPE exp_26 CHANGES (rgroup 1 -> 2)"),
    ((8192, 8192, 29568, False), 40, "row 6, 4 tiles/CTA, unchanged control"),
    ((4096, 4096, 4096, False), 30, "row 4, 1 tile/CTA, unchanged control"),
]
M9.CONTROL_EPOCHS = [30, 20, 10]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0], sys.argv[1]]
    else:
        sys.argv = [sys.argv[0]]
    sys.exit(M9.main())
