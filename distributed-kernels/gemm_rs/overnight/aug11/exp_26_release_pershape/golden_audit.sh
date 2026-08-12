#!/usr/bin/env bash
# Is m9's frozen golden still a valid comparand, or does it carry a superseded
# shape table? CPU-only.
#
# Hypothesis under test: exp_14 (E4b) retiled rows 1, 2 and 3 AFTER the golden
# copy was taken. For row 1 the change is BK-only (64 -> 128), which leaves the
# tile map and the signal geometry identical, so the golden stays bit-exact. For
# rows 2 and 3 it changed BN, which changes col_count and therefore the whole
# tile -> (dest, lrow, col) map the harness allocates signals for -- and row 3's
# old BN=256 against N=2880 is exactly the out-of-bounds B read exp_14's own
# comment says the retile removed.
#
# If that is what the tables say, then m9's row-2 failure and the 10:05Z memory
# access fault on row 3 are properties of the GOLDEN, not of the candidate.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
BASE=$ON/experiments/exp_05_release_granularity/baseline

echo "########## golden source provenance ##########"
ls -la "$BASE"/*.cpp "$BASE"/*.hpp 2>/dev/null
echo
echo "-- which host_abi does the golden compile against? --"
grep -nE 'gemm_rs_mi300x_host_abi|include "' "$BASE/gemm_rs_mi300x_e3base.cpp" 2>/dev/null | head
echo
echo "########## the golden's dispatch table ##########"
grep -nE 'case [0-9]+: launch_fixed' "$BASE/gemm_rs_mi300x_e3base.cpp" 2>/dev/null
echo
echo "########## the CURRENT dispatch table ##########"
grep -nE 'case [0-9]+: launch_fixed' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x.cpp
echo
echo "########## the scored table the harness resolves against (current) ##########"
sed -n '/inline constexpr std::array<shape_entry, 6> scored_shapes/,/}};/p' \
  /home/subvadla/dhk/distributed-kernels/gemm_rs/gemm_rs_mi300x_host_abi.hpp
echo
echo "########## implied geometry per row, golden vs current ##########"
python3 - <<'PY'
rows = [
    # m, n, name, (golden bm,bn,bk), (current bm,bn,bk)
    (  64, 7168, "row 1", (32,  64,  64), (32,  64, 128)),
    ( 512, 4096, "row 2", (64,  64,  64), (64, 128,  64)),
    (2048, 2880, "row 3", (128,256,  32), (128,192,  32)),
    (4096, 4096, "row 4", (256,256,  32), (256,256,  32)),
    (8192, 4096, "row 5", (256,256,  32), (256,256,  32)),
    (8192, 8192, "row 6", (256,256,  32), (256,256,  32)),
]
print(f"{'row':<7}{'N':>6} | {'golden BM/BN/BK':>16}{'cols':>6}{'tiles':>7}{'B rows read':>13}"
      f" | {'current BM/BN/BK':>17}{'cols':>6}{'tiles':>7}{'B rows read':>13}  verdict")
for m, n, name, g, c in rows:
    out = f"{name:<7}{n:>6} | "
    cells = []
    for bm, bn, bk in (g, c):
        cols = -(-n // bn)
        tiles = (m // bm) * cols
        maxrow = cols * bn          # last B tile ends here
        cells.append((f"{bm}/{bn}/{bk}", cols, tiles, maxrow))
    for tag, cols, tiles, maxrow in cells:
        oob = "" if maxrow <= n else f" OOB+{maxrow - n}"
        out += f"{tag:>16}{cols:>6}{tiles:>7}{str(maxrow) + oob:>13} | "
    same = cells[0][1] == cells[1][1]
    out += ("map identical" if same else "MAP DIFFERS -> golden is not a valid comparand")
    print(out)
PY
