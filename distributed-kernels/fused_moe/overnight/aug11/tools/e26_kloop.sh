#!/usr/bin/env bash
# exp_26 step 3d: locate the two MFMA K-loops by backedge, census them, and
# dump the phase-1 loop schedule for B1 vs B2.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
mkdir -p "$SC/py"

cat > "$SC/py/kloop.py" <<'PYEOF'
import re, sys, collections

def load(path):
    insns, base = [], None
    for line in open(path, errors='replace'):
        m = re.match(r'^([0-9a-f]{16}) <([^>]+)>:', line)
        if m:
            base = int(m.group(1), 16); continue
        m = re.match(r'^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):\s*(.*)$', line.rstrip('\n'))
        if not m: continue
        mnem, ops, addr, tail = m.group(1), m.group(2), int(m.group(3),16), m.group(4)
        tgt = None
        mt = re.search(r'<[^>]*\+0x([0-9a-f]+)>', tail)
        if mt and base is not None: tgt = base + int(mt.group(1),16)
        insns.append((addr, mnem, ops, tgt))
    return insns

def cls(m):
    if m.startswith('v_mfma'): return 'MFMA'
    if m.startswith('ds_read'): return 'DSR'
    if m.startswith('ds_write'): return 'DSW'
    if m.startswith('scratch_'): return 'SCR'
    if m.startswith(('buffer_load','global_load','flat_load')): return 'VMLD'
    if m.startswith(('buffer_store','global_store','flat_store','buffer_atomic','global_atomic','flat_atomic')): return 'VMST'
    if m == 's_waitcnt': return 'WAIT'
    if m.startswith('s_barrier'): return 'BAR'
    if m.startswith('v_'): return 'VALU'
    if m.startswith('s_'): return 'SALU'
    return 'oth'

def loops(insns):
    out = []
    for i,(a,m,o,t) in enumerate(insns):
        if t is not None and m.startswith(('s_cbranch','s_branch')) and t <= a:
            out.append((t,a))
    return out

def body(insns, lo, hi):
    return [x for x in insns if lo <= x[0] <= hi]

def census(b):
    c = collections.Counter(cls(m) for _,m,_,_ in b)
    return c

def rle(seq):
    out=[]
    for s in seq:
        if out and out[-1][0]==s: out[-1][1]+=1
        else: out.append([s,1])
    return ' '.join(f'{s}x{n}' if n>1 else s for s,n in out)

def waits(b):
    return [o for _,m,o,_ in b if m=='s_waitcnt']

def report(tag, path):
    insns = load(path)
    ls = loops(insns)
    found = {}
    for lo,hi in ls:
        b = body(insns, lo, hi)
        n = census(b)['MFMA']
        if n >= 40:
            found.setdefault(n, []).append((lo,hi,b))
    print(f'===== {tag} =====')
    print(f'  total insns={len(insns)} backedges={len(ls)}')
    for n in sorted(found, reverse=True):
        for lo,hi,b in found[n]:
            c = census(b)
            print(f'  LOOP mfma={n} range=0x{lo:x}..0x{hi:x} insns={len(b)} '
                  f'DSR={c["DSR"]} DSW={c["DSW"]} VMLD={c["VMLD"]} VMST={c["VMST"]} '
                  f'SCR={c["SCR"]} WAIT={c["WAIT"]} BAR={c["BAR"]} VALU={c["VALU"]} SALU={c["SALU"]}')
    return found

def dump_p1(tag, path, out):
    insns = load(path)
    for lo,hi in loops(insns):
        b = body(insns, lo, hi)
        if census(b)['MFMA'] == 96:
            bars = [i for i,(a,m,o,t) in enumerate(b) if m.startswith('s_barrier')]
            print(f'--- {tag} phase-1 K-loop 0x{lo:x}..0x{hi:x} insns={len(b)} barriers_at={bars}')
            # split into halves at the barriers
            segs, prev = [], 0
            for bi in bars:
                segs.append(b[prev:bi+1]); prev = bi+1
            if prev < len(b): segs.append(b[prev:])
            for si,seg in enumerate(segs):
                c = census(seg)
                print(f'  SEG{si}: insns={len(seg)} MFMA={c["MFMA"]} DSR={c["DSR"]} DSW={c["DSW"]} '
                      f'VMLD={c["VMLD"]} SCR={c["SCR"]} WAIT={c["WAIT"]}')
                print(f'    waits: {waits(seg)}')
                print(f'    seq  : {rle([cls(m) for _,m,_,_ in seg])}')
            with open(out,'w') as f:
                for a,m,o,t in b: f.write(f'{a:08X}\t{m}\t{o}\n')
            return
    print(f'--- {tag}: NO 96-MFMA loop found')

for tag in sys.argv[1:]:
    report(tag, f'{tag}.isa')
print()
for tag in sys.argv[1:]:
    dump_p1(tag, f'{tag}.isa', f'{tag}.p1loop.txt')
PYEOF

docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out && python3 ../py/kloop.py B0 B1 B2"
