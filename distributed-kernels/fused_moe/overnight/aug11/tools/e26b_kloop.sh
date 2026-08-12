#!/usr/bin/env bash
# exp_26 follow-up step 3: per-mask K-loop schedule + scratch placement.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
mkdir -p "$SC/py"

cat > "$SC/py/mask.py" <<'PYEOF'
import re, sys, collections

def load(path):
    insns, base, fn = [], None, None
    for line in open(path, errors='replace'):
        m = re.match(r'^([0-9a-f]{16}) <([^>]+)>:', line)
        if m:
            base = int(m.group(1), 16); fn = m.group(2); continue
        m = re.match(r'^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):\s*(.*)$', line.rstrip('\n'))
        if not m: continue
        mnem, ops, addr, tail = m.group(1), m.group(2), int(m.group(3),16), m.group(4)
        tgt = None
        mt = re.search(r'<[^>]*\+0x([0-9a-f]+)>', tail)
        if mt and base is not None: tgt = base + int(mt.group(1),16)
        insns.append((addr, mnem, ops, tgt, fn))
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

def backedges(insns):
    return [(t,a) for (a,m,o,t,f) in insns
            if t is not None and m.startswith(('s_cbranch','s_branch')) and t <= a]

def body(insns, lo, hi):
    return [x for x in insns if lo <= x[0] <= hi]

def find_loop(insns, nmfma):
    best = None
    for lo,hi in backedges(insns):
        b = body(insns, lo, hi)
        if sum(1 for x in b if cls(x[1])=='MFMA') == nmfma:
            if best is None or (hi-lo) < (best[1]-best[0]): best = (lo,hi)
    return best

def kloop_report(tag, insns):
    lo, hi = find_loop(insns, 96)
    b = body(insns, lo, hi)
    # walk, tracking mfma index; note every barrier and every vmcnt-bearing wait
    n, events, segs, seg = 0, [], [], 0
    for (a,m,o,t,f) in b:
        c = cls(m)
        if c == 'MFMA': n += 1
        elif c == 'BAR':
            events.append(('barrier', n)); segs.append(n - seg); seg = n
        elif c == 'WAIT':
            mv = re.search(r'vmcnt\((\d+)\)', o)
            if mv: events.append((f'vmcnt({mv.group(1)}){"+lgkm" if "lgkmcnt" in o else ""}', n))
    segs.append(n - seg)
    c = collections.Counter(cls(x[1]) for x in b)
    return dict(lo=lo, hi=hi, insns=len(b), segs=segs, events=events,
                DSR=c['DSR'], DSW=c['DSW'], VMLD=c['VMLD'], SCR=c['SCR'], WAIT=c['WAIT'])

def scratch_split(insns):
    p1 = find_loop(insns, 96)
    p2 = find_loop(insns, 84)
    buckets = collections.Counter()
    for (a,m,o,t,f) in insns:
        if cls(m) != 'SCR': continue
        if a < p1[0]:                 buckets['pre_p1'] += 1
        elif a <= p1[1]:              buckets['in_p1'] += 1
        elif a < p2[0]:               buckets['between'] += 1
        elif a <= p2[1]:              buckets['in_p2'] += 1
        else:                         buckets['after_p2'] += 1
    return p1, p2, buckets

rows = []
for tag in sys.argv[1:]:
    insns = load(f'{tag}.isa')
    k = kloop_report(tag, insns)
    p1, p2, sb = scratch_split(insns)
    tot = sum(sb.values())
    before = sb['pre_p1']+sb['in_p1']+sb['between']+sb['in_p2']
    rows.append((tag, k, p1, p2, sb, tot, before))
    print(f'===== {tag} =====')
    print(f'  total insns={len(insns)}  MFMA={sum(1 for x in insns if cls(x[1])=="MFMA")}')
    print(f'  p1 K-loop 0x{p1[0]:x}..0x{p1[1]:x} ({k["insns"]} insns)  '
          f'DSR={k["DSR"]} DSW={k["DSW"]} VMLD={k["VMLD"]} SCR={k["SCR"]} WAIT={k["WAIT"]}')
    print(f'  p2 K-loop 0x{p2[0]:x}..0x{p2[1]:x}')
    print(f'  barrier partition (MFMAs per segment): {k["segs"]}')
    print(f'  events (kind @ mfma_index): {k["events"]}')
    print(f'  scratch total={tot}  pre_p1={sb["pre_p1"]} in_p1={sb["in_p1"]} '
          f'between={sb["between"]} in_p2={sb["in_p2"]} AFTER_p2={sb["after_p2"]}')

print()
print('mask | scr_total | <=p2 | after_p2 | drain@mfma | partition')
for tag,k,p1,p2,sb,tot,before in rows:
    dr = [i for (e,i) in k['events'] if e.startswith('vmcnt(0)')]
    print(f'{tag:4s} | {tot:9d} | {before:4d} | {sb["after_p2"]:8d} | {str(dr):10s} | {k["segs"]}')
PYEOF

docker exec subha_k1 bash -lc "cd /home/subvadla/overnight-scratch/e26/out2 && python3 ../py/mask.py D M0 M2 M4 M6 M9 M15"
