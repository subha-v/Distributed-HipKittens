#!/usr/bin/env bash
# exp_24 dry-run validation. NO GPU. Three things:
#   1. every file compiles / parses (py_compile + bash -n), so a syntax error is
#      not discovered 90 minutes into a ladder;
#   2. the popcorn parser runs against the three SAVED historical evaluator
#      outputs under compbench/ -- the cheapest honest way to validate a text
#      parser without burning a GPU run;
#   3. run_ladders.sh's preflight is exercised, which on a dirty node must ABORT.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders

echo "=== 1a. bash -n ==="
for f in "$D"/*.sh; do bash -n "$f" && echo "OK  $(basename "$f")" || echo "FAIL $(basename "$f")"; done

echo
echo "=== 1b. python syntax ==="
docker exec dhk-gemmrs bash -lc "cd $D && python3 -m py_compile ladders.py ladder_mp.py && echo 'OK  both compile'" 2>&1 | tail -5

echo
echo "=== 1c. ladder_mp.py imports + arm/rotation tables, no torch.distributed ==="
docker exec dhk-gemmrs bash -lc "cd $D && python3 -c \"
import ladder_mp as L
arms = [s[0] for s in L.ARM_SPECS]
print('arms      :', arms)
print('ungated   :', sorted(L.UNGATED))
print('non-ratio :', sorted(L.NON_RATIO))
# ours and ours_null MUST be the same file under different module names.
by = {s[0]: s for s in L.ARM_SPECS}
assert by['ours'][2] == by['ours_null'][2], 'null arm is not the same file'
assert by['ours'][1] != by['ours_null'][1], 'null arm shares a module name'
print('null arm  : same file, distinct module names -- OK')
firsts = []
for rep in range(len(arms)):
    sh = rep % len(arms)
    order = arms[sh:] + arms[:sh]
    protos = ('graded','pipelined') if rep%2==0 else ('pipelined','graded')
    firsts.append((order[0], protos[0]))
print('rep -> (first arm, first protocol):')
for i,f in enumerate(firsts): print('   rep', i, f)
assert len(set(a for a,_ in firsts)) == len(arms), 'an arm is never first'
print('rotation  : every arm first exactly once -- OK')
print('shapes    :', [f'{m}x{n}x{k}' for m,n,k,_,_ in L.SCORED])
\"" 2>&1 | tail -25

echo
echo "=== 2. THE FIXTURE TEST: parse the saved historical popcorn output ==="
docker exec dhk-gemmrs bash -lc "cd $D && python3 ladders.py --fixture" 2>&1
echo "fixture rc=$?"

echo
echo "=== 2b. negative control: the parser MUST reject a test-mode popcorn file ==="
docker exec dhk-gemmrs bash -lc "cd $D && python3 ladders.py --fixture $ON/compbench/ours/test.popcorn.txt" 2>&1 | tail -4

echo
echo "=== 2c. negative control: MUST reject a truncated benchmark file ==="
docker exec dhk-gemmrs bash -lc "mkdir -p /tmp/lad_fx/trunc && head -14 $ON/compbench/ours/benchmark.popcorn.txt > /tmp/lad_fx/trunc/benchmark.popcorn.txt && cd $D && python3 ladders.py --fixture /tmp/lad_fx/trunc/benchmark.popcorn.txt" 2>&1 | tail -4

echo
echo "=== 2d. negative control: aggregator MUST refuse an empty ladder ==="
docker exec dhk-gemmrs bash -lc "mkdir -p /tmp/lad_empty && cd $D && python3 ladders.py --root /tmp/lad_empty --out /tmp/lad_empty/ladders.json" 2>&1 | tail -4
echo "(rc must be nonzero above)"

echo
echo "=== 3. run_ladders.sh preflight on a DIRTY node: must abort ==="
bash "$D/run_ladders.sh" preflight 2>&1 | tail -20
echo "preflight rc=$? (nonzero expected while another job holds the GPUs)"

echo
echo "=== 4. staging paths one more time, for the record ==="
sha256sum /home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
echo "expected 7940fcb81df06c1d8b1e1a77051f23c934149a688441ef48b2751b3f336f0dc5"
ls "$ON/tools/patch_rank1.py" "$ON/experiments/exp_10_rank1/r1_eval.sh" 2>&1

echo
echo "=== VALIDATION DONE ==="
