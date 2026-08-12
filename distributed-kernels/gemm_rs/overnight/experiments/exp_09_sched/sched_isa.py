#!/usr/bin/env python3
# exp_09_sched: schedule-shape reader for the producer k-loop.
#
#   sched_isa.py <isa.s> [instantiation-prefix] > report.txt
#
# Same loop-discovery method as experiments/exp_03_mainloop/kloop_isa.py (find
# the deepest loop that contains v_mfma via LLVM's `Depth=` block annotations,
# so it survives source edits that move line numbers), but reports the ONE
# thing this experiment is about: the program order of the schedule-relevant
# instruction classes across the whole loop body, with basic-block boundaries
# marked.
#
# Why block boundaries matter here: __builtin_amdgcn_sched_group_barrier only
# reorders inside one scheduling region, and a region never spans two basic
# blocks. If the global loads sit in their own block, no sched builtin can
# interleave them with the MFMAs.
import re, sys, os, collections

s_path = sys.argv[1]
want   = sys.argv[2] if len(sys.argv) > 2 else '<256,256,32'
lines  = open(s_path, errors='replace').read().splitlines()

re_fstart = re.compile(r'^(_Z\S+):\s*(;.*)?$')
re_fend   = re.compile(r'^\.Lfunc_end(\d+):')
funcs, cur = [], None
for i, L in enumerate(lines, 1):
    m = re_fstart.match(L)
    if m: cur = (m.group(1), i); continue
    if re_fend.match(L) and cur: funcs.append([cur[0], cur[1], i]); cur = None

def tag(n):
    nu = re.findall(r'Li(\d+)E', n); tl = re.search(r'Lb(\d)E', n)
    return "<%s,%s,%s,%s>" % (nu[0], nu[1], nu[2],
        'true' if tl and tl.group(1) == '1' else 'false') if len(nu) >= 3 else n

re_blk    = re.compile(r'^(\.LBB[\w.$]+):|^; (%bb\.[\w.$]+):')
re_inloop = re.compile(r'in Loop: Header=(BB[\w.$]+) Depth=(\d+)')
re_hdr    = re.compile(r'=>\s*This (Loop|Inner Loop) Header: Depth=(\d+)')

def blocks_of(a, b):
    out = []
    for i in range(a, b + 1):
        m = re_blk.match(lines[i - 1])
        if not m: continue
        ann = lines[i - 1]
        if i < len(lines):     ann += " || " + lines[i]
        if i + 1 < len(lines): ann += " || " + lines[i + 1]
        h, hd = re_inloop.search(ann), re_hdr.search(ann)
        out.append(dict(name=(m.group(1) or m.group(2)), start=i,
                        hdr=(h.group(1) if h else None),
                        depth=int(h.group(2)) if h else (int(hd.group(2)) if hd else 0),
                        is_hdr=bool(hd)))
    for k in range(len(out)):
        out[k]['end'] = (out[k + 1]['start'] - 1) if k + 1 < len(out) else b
    return out

def instrs(a, b):
    o, inasm = [], False
    for i in range(a, b + 1):
        raw = lines[i - 1]
        if ';;#ASMSTART' in raw: inasm = True;  continue
        if ';;#ASMEND'   in raw: inasm = False; continue
        st = raw.split(';')[0].strip()
        if not st or st.startswith('.') or st.endswith(':'): continue
        o.append((i, st.split()[0], st, inasm))
    return o

SYNC = ('global_load', 'global_store', 'buffer_load', 'buffer_store',
        'buffer_wbl2', 'buffer_inv', 'ds_read', 'ds_write', 's_waitcnt',
        's_barrier', 'v_mfma', 's_setprio', 'scratch_store', 'scratch_load',
        's_sched', 'sched_')
def issync(m): return any(m.startswith(p) for p in SYNC)

for nm, a, b in funcs:
    t = tag(nm)
    if not t.startswith(want): continue
    blks = blocks_of(a, b)
    byhdr = collections.defaultdict(list)
    for bl in blks:
        if bl['hdr']: byhdr[(bl['hdr'], bl['depth'])].append(bl)
    hdrblk = {bl['name'].replace('.L', ''): bl for bl in blks if bl['is_hdr']}
    # A single-block loop has no member carrying an `in Loop: Header=` comment
    # -- the one block IS the loop and only says "This Inner Loop Header" --
    # so seed a group from every header. Without this the reader reported
    # "NO MFMA LOOP FOUND" on exactly the arm that succeeded in collapsing the
    # k-loop to one block.
    for key, bl in hdrblk.items():
        byhdr.setdefault((key, bl['depth']), []).append(bl)
    best = None
    for (h, d), bs in sorted(byhdr.items(), key=lambda kv: -kv[0][1]):
        allb = list(bs)
        hb = hdrblk.get(h)
        if hb and hb not in allb: allb.append(hb)
        allb.sort(key=lambda x: x['start'])
        nmf = sum(1 for bl in allb for x in instrs(bl['start'], bl['end'])
                  if x[1].startswith('v_mfma'))
        if nmf and (best is None or d > best[1]): best = (h, d, allb, nmf)
    print("=" * 100)
    print("SYMBOL %s  %s" % (t, nm))
    if not best:
        print("  NO MFMA LOOP FOUND"); continue
    h, d, allb, nmf = best
    print("  deepest MFMA loop Header=%s Depth=%d  blocks=%d  mfma=%d  lines %d..%d"
          % (h, d, len(allb), nmf, allb[0]['start'], allb[-1]['end']))
    tot = collections.Counter()
    print("  BLOCKS:")
    for bl in allb:
        ops = instrs(bl['start'], bl['end'])
        term = [x[2] for x in ops if x[1].startswith('s_cbranch')
                or x[1] in ('s_branch', 's_endpgm')]
        print("    %-14s %6d..%-6d %4d instr  term=%s"
              % (bl['name'], bl['start'], bl['end'], len(ops),
                 (term[-1] if term else 'fallthrough')))
    print("  PROGRAM ORDER (run-length; A=inline asm, C=compiler):")
    for bl in allb:
        print("    -- %s --" % bl['name'])
        run = []
        for (i, mn, st, ia) in instrs(bl['start'], bl['end']):
            tot[mn] += 1
            if not issync(mn): continue
            key = (st if mn == 's_waitcnt' else mn, 'A' if ia else 'C')
            if run and run[-1][0] == key: run[-1][1] += 1; run[-1][3] = i
            else: run.append([key, 1, i, i])
        for (key, n, i0, i1) in run:
            print("       [%s] x%-4d %-38s L%d..L%d" % (key[1], n, key[0], i0, i1))
    print("  TOTALS in loop: " + "  ".join(
        "%s=%d" % (k, v) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])
        if issync(k)))
