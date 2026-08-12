#!/usr/bin/env bash
# exp_26 follow-up step 5: (a) the full s_waitcnt list inside phase 1's K-loop
# for the four distinct behaviour classes, (b) where the migrated scratch
# accesses actually land, (c) whether bit 0 perturbs PHASE 2's K-loop too.
set -uo pipefail
SC=$HOME/overnight-scratch/e26

cat > "$SC/py/detail.py" <<'PYEOF'
import re, sys, collections
exec(open('/home/subvadla/overnight-scratch/e26/py/mask.py').read().split('rows = []')[0])

def fn_extent(insns):
    return min(a for a,_,_,_,_ in insns), max(a for a,_,_,_,_ in insns)

for tag in sys.argv[1:]:
    insns = load(f'{tag}.isa')
    p1 = find_loop(insns, 96); p2 = find_loop(insns, 84)
    lo_f, hi_f = fn_extent(insns)
    print(f'===== {tag} =====')
    print(f'  fn 0x{lo_f:x}..0x{hi_f:x}   p1 0x{p1[0]:x}..0x{p1[1]:x}   p2 0x{p2[0]:x}..0x{p2[1]:x}')

    for name, rng, nm in (('P1', p1, 96), ('P2', p2, 84)):
        b = body(insns, *rng)
        n, ev = 0, []
        for (a,m,o,t,f) in b:
            c = cls(m)
            if c == 'MFMA': n += 1
            elif c == 'BAR': ev.append(f'BARRIER@{n}')
            elif c == 'WAIT': ev.append(f'{o.strip()}@{n}')
        segs, prev = [], 0
        nn = 0
        for (a,m,o,t,f) in b:
            if cls(m)=='MFMA': nn += 1
            elif cls(m)=='BAR': segs.append(nn-prev); prev = nn
        segs.append(nn-prev)
        print(f'  {name} loop mfma={nm} partition={segs}')
        print(f'     {" | ".join(ev)}')

    # where the after-p2 scratch lands, as a fraction of the post-p2 tail
    tail = [a for a,m,_,_,_ in insns if cls(m)=='SCR' and a > p2[1]]
    if tail:
        span = hi_f - p2[1]
        buckets = collections.Counter(int(10*(a-p2[1])/max(span,1)) for a in tail)
        print(f'  after-p2 scratch: n={len(tail)} first=0x{min(tail):x} last=0x{max(tail):x}')
        print(f'     decile histogram over the post-p2 tail: '
              + ' '.join(f'{d}:{buckets.get(d,0)}' for d in range(10)))
        # is any of it inside a loop?
        inloop = 0
        for blo,bhi in backedges(insns):
            if blo > p2[1]:
                inloop += sum(1 for a in tail if blo <= a <= bhi)
        print(f'     accesses inside some post-p2 backedge (double counted): {inloop}')
PYEOF

docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out2 && python3 ../py/detail.py D M4 M1 M15"
