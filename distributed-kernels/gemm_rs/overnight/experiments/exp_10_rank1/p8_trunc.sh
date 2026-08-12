#!/usr/bin/env bash
# exp_10 probe 8 -- NO GPU. Confirm in the compiled IR that the heap_base
# constexpr offsets are narrowed to i32, and read the kernel's peer-store path
# to see whether a faithful repair exists.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
CACHE=$ARM/.triton
RANK1=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py

echo "===== A. which cache entry is the shape-1 GEMM? (look at constants) ====="
for d in "$CACHE"/*/; do
  j="$d/_kernel1.json"
  [ -f "$j" ] || continue
  echo "--- $(basename "$d") ---"
  python3 -c "
import json,sys
j=json.load(open('$j'))
print('  name    :', j.get('name'))
print('  n_regs  :', j.get('n_regs'), 'spills:', j.get('n_spills'), 'shared:', j.get('shared'))
c=j.get('constants') or {}
hb={k:v for k,v in c.items() if 'heap' in str(k).lower()}
print('  heap consts:', hb if hb else '(none in json)')
print('  M,N,K   :', {k:v for k,v in c.items() if str(k) in ('M','N','K')})
" 2>/dev/null || echo "  (json parse failed)"
done

echo
echo "===== B. grep the IR for the heap_base literals and their TYPES ====="
# The base_addrs seen at runtime, exact and int32-truncated.
for V in -3850813952 444153344 -4388733440 -93766144 -3312894464 982072832; do
  echo "--- literal $V ---"
  grep -rl -- "$V" "$CACHE" 2>/dev/null | head -4 | while read -r f; do
    echo "  in $(basename "$(dirname "$f")")/$(basename "$f"):"
    grep -oE "[^ ]*$V[^ ]*" "$f" 2>/dev/null | sort -u | head -4 | sed 's/^/      /'
  done
done

echo
echo "===== C. in the ttir/llir: are the heap offsets i32 or i64? ====="
TT=$(find "$CACHE" -name '_kernel1.ttir' | head -1)
echo "using $TT"
echo "--- addptr ops and their offset types ---"
grep -nE 'addptr|arith.constant.*i(32|64)' "$TT" 2>/dev/null | grep -iE 'i32|i64' | head -30

echo
echo "--- the .llir: getelementptr with sext/zext, and any trunc to i32 ---"
LL=$(find "$CACHE" -name '_kernel1.llir' | head -1)
echo "using $LL"
grep -cE 'getelementptr' "$LL" 2>/dev/null
echo "trunc-to-i32 count: $(grep -cE 'trunc i64 .* to i32' "$LL" 2>/dev/null)"
echo "sext-i32-to-i64 count: $(grep -cE 'sext i32 .* to i64' "$LL" 2>/dev/null)"

echo
echo "===== D. _kernel1: the peer store / epilogue path (1250..1290 signature) ====="
awk 'NR>=1250 && NR<=1300 {printf "%d: %s\n", NR, $0}' "$RANK1"

echo
echo "===== E. where heap_base_* is USED in the kernel body ====="
grep -nE 'heap_base|my_rank_base|A_index|a_ptr|tl\.store|store_to|noti|NOTI' "$RANK1" \
  | awk -F: '$1>=1240 && $1<=1380'

echo
echo "===== F. how launch_triton_kernel builds the call (1436..1500) ====="
awk 'NR>=1436 && NR<=1500 {printf "%d: %s\n", NR, $0}' "$RANK1"
echo "===== DONE p8 ====="
