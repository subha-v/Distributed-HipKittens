#!/usr/bin/env bash
# exp_24 dry-run probe: READ-ONLY. Touches no GPU compute, writes nothing
# outside $D. Establishes that all three arms can be staged, that the frozen
# rank-1 submission still hashes to the expected value, and locates a saved
# historical popcorn/stdout output to use as a parser fixture.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
EXPECT=7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5

echo "=== host $(hostname) $(date -Is) ==="

echo
echo "=== 1. containers ==="
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' 2>&1 | sed -n '1,20p'

echo
echo "=== 2. frozen rank-1 submission (read-only hash) ==="
if [ -r "$RANK1" ]; then
  GOT=$(sha256sum "$RANK1" | awk '{print $1}')
  echo "path   : $RANK1"
  echo "sha256 : $GOT"
  echo "expect : $EXPECT"
  [ "$GOT" = "$EXPECT" ] && echo "VERDICT: HASH MATCH" || echo "VERDICT: HASH MISMATCH"
  echo "perms  : $(ls -l "$RANK1" | awk '{print $1, $3, $5}')"
else
  echo "VERDICT: NOT READABLE at $RANK1"
fi

echo
echo "=== 3. patch_rank1.py locations (bench3 calls \$NODE/patch_rank1.py) ==="
for p in "$ON/patch_rank1.py" "$ON/tools/patch_rank1.py"; do
  if [ -f "$p" ]; then echo "PRESENT $p  ($(sha256sum "$p" | cut -c1-12))"; else echo "MISSING $p"; fi
done

echo
echo "=== 4. evaluator source tree (all three arms stage from here) ==="
for f in eval.py task.py utils.py reference.py submission.py cases.txt; do
  if [ -f "$SRC/$f" ]; then echo "PRESENT $SRC/$f  $(wc -l < "$SRC/$f") lines"; else echo "MISSING $SRC/$f"; fi
done

echo
echo "=== 5. our built modules (ours arm) ==="
ls -l --time-style=full-iso "$ON/harness/build/" 2>&1 | sed -n '1,12p'
echo -n "submission.py == hk_submission.py: "
cmp -s "$ON/harness/submission.py" "$ON/harness/hk_submission.py" && echo yes || echo NO

echo
echo "=== 6. iris staging, per container (separate filesystems!) ==="
for c in dhk-gemmrs dhk-eval; do
  echo "--- $c ---"
  docker exec "$c" bash -lc 'ls -d /usr/local/lib/python3.10/dist-packages/iris 2>/dev/null && python3 -c "import sys;sys.path.insert(0,\"/usr/local/lib/python3.10/dist-packages\");import iris,iris.hip;print(\"import iris OK\", iris.__file__)" 2>&1 | tail -2' 2>&1 | sed -n '1,6p'
done

echo
echo "=== 7. compat shims (sudo no-op + sitecustomize) ==="
ls -l "$ON/tools/compat/bin/sudo" "$ON/tools/compat/sitecustomize.py" 2>&1
ls -d "$ON/compat" 2>&1

echo
echo "=== 8. saved historical evaluator outputs (parser fixtures) ==="
find "$ON/compbench" "$ON/experiments" "$ON/logs" -maxdepth 4 \
     \( -name '*popcorn*.txt' -o -name '*.stdout.txt' \) -size +0 2>/dev/null \
  | while read -r f; do echo "$(wc -c < "$f") $f"; done | sort -rn | sed -n '1,40p'

echo
echo "=== 9. exp_10 raw sample JSONs (interleaved-instrument fixtures) ==="
ls -l "$ON/experiments/exp_10_rank1/vs_logs/" 2>&1 | sed -n '1,8p'
ls "$ON/experiments/exp_14_tile_waves/vs_logs/" 2>&1 | sed -n '1,8p'

echo
echo "=== 10. node state: KFD pids + clocks (read-only) ==="
rocm-smi --showpids 2>&1 | sed -n '1,25p'
rocm-smi --showclocks 2>&1 | grep -iE 'sclk|GPU\[0\]' | sed -n '1,6p'

echo
echo "=== 11. current scored_shapes table (the config under test) ==="
sed -n '/inline constexpr std::array<shape_entry, 6> scored_shapes/,/}};/p' \
  "$ON/../gemm_rs_mi300x_host_abi.hpp"

echo
echo "=== 12. constants that define the exp_23 winner config ==="
grep -nE 'RELEASE_GROUP|NUM_REDUCER_CTAS|WGM|WG_M' "$ON/../gemm_rs_mi300x_constants.cuh" 2>&1 | sed -n '1,25p'

echo
echo "=== PROBE DONE ==="
