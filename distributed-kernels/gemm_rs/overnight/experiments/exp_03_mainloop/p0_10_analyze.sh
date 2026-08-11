#!/usr/bin/env bash
# P0 step 1: structural analysis of the already-built gfx942 ISA.
# READ-ONLY. No GPU work, no build, no harness.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s
mkdir -p "$EXP/logs" "$EXP/isa"

echo "ISA: $S"
ls -la "$S"; wc -l "$S"

cat > "$EXP/analyze_isa.py" <<'PYEOF'
import re, sys, os, collections

s_path, outdir = sys.argv[1], sys.argv[2]
lines = open(s_path, errors='replace').read().splitlines()
LOG = os.path.join(outdir, 'logs')
ISA = os.path.join(outdir, 'isa')

# ---------------- 1. function regions ----------------
re_fstart = re.compile(r'^(_Z\S+):\s*(;.*)?$')
re_fend   = re.compile(r'^\.Lfunc_end(\d+):')
funcs, cur = [], None
for i, L in enumerate(lines, 1):
    m = re_fstart.match(L)
    if m:
        cur = (m.group(1), i); continue
    if re_fend.match(L) and cur:
        funcs.append([cur[0], cur[1], i]); cur = None

def tag(name):
    nums = re.findall(r'Li(\d+)E', name)
    tl   = re.search(r'Lb(\d)E', name)
    if len(nums) >= 3:
        return "<%s,%s,%s,%s>" % (nums[0], nums[1], nums[2],
                                  'true' if (tl and tl.group(1)=='1') else 'false')
    return '(non-template)'

# ---------------- 2. metadata YAML ----------------
txt = "\n".join(lines)
blocks = re.findall(r'- \.agpr_count:.*?(?=\n  - \.agpr_count:|\namdhsa\.|\Z)', txt, re.S)
meta = {}
for b in blocks:
    def g(k, d='-'):
        m = re.search(r'\.%s:\s*(\S+)' % k, b); return m.group(1) if m else d
    nm = g('name')
    meta[nm] = dict(vgpr=g('vgpr_count'), agpr=g('agpr_count'), sgpr=g('sgpr_count'),
                    scratch=g('private_segment_fixed_size'), lds=g('group_segment_fixed_size'),
                    maxwg=g('max_flat_workgroup_size'), spill_v=g('vgpr_spill_count'),
                    spill_s=g('sgpr_spill_count'), waveflat=g('wavefront_size'))

# .amdhsa_kernel directive blocks (authoritative encoding fields)
adir = {}
i = 0
while i < len(lines):
    m = re.match(r'^\s*\.amdhsa_kernel\s+(\S+)', lines[i])
    if m:
        nm, d = m.group(1), {}
        j = i+1
        while j < len(lines) and '.end_amdhsa_kernel' not in lines[j]:
            mm = re.match(r'^\s*\.amdhsa_(\S+)\s+(\S+)', lines[j])
            if mm: d[mm.group(1)] = mm.group(2)
            j += 1
        adir[nm] = d; i = j
    i += 1

# ---------------- 3. instruction helper ----------------
def instrs(a, b):
    out = []
    for i in range(a, b+1):
        code = lines[i-1].split(';')[0].rstrip()
        st = code.strip()
        if not st or st.startswith('.') or st.endswith(':'):
            continue
        out.append((i, st.split()[0], st))
    return out

# ---------------- 4. loop detection ----------------
re_label  = re.compile(r'^(\.LBB[\w.$]+):')
re_branch = re.compile(r'^\s+(s_cbranch_\w+|s_branch)\s+(\.LBB[\w.$]+)')

def loops_of(a, b):
    labels = {}
    for i in range(a, b+1):
        m = re_label.match(lines[i-1])
        if m: labels[m.group(1)] = i
    edges = []
    for i in range(a, b+1):
        m = re_branch.match(lines[i-1])
        if m:
            t = labels.get(m.group(2))
            if t is not None and t < i:
                edges.append((t, i, m.group(1), m.group(2)))
    return sorted(set(edges))

SYNC = ('global_load','global_store','global_atomic','buffer_load','buffer_store',
        'buffer_wbl2','buffer_inv','buffer_atomic','flat_load','flat_store','flat_atomic',
        'ds_read','ds_write','ds_add','s_waitcnt','s_barrier','v_mfma','s_setprio',
        'scratch_store','scratch_load','s_sleep','s_load','s_memrealtime','s_endpgm')
def is_sync(mn):
    return any(mn.startswith(p) for p in SYNC)

def waitargs(st):
    return st.replace('s_waitcnt','').strip()

def seq(ops):
    """run-length encode program order of sync-relevant ops"""
    out = []
    for (i, mn, st) in ops:
        if not is_sync(mn): continue
        if mn == 's_waitcnt':
            key = 's_waitcnt ' + waitargs(st)
        elif mn.startswith('v_mfma'):
            key = mn
        else:
            key = mn
        if out and out[-1][0] == key:
            out[-1][1] += 1; out[-1][3] = i
        else:
            out.append([key, 1, i, i])
    return out

# ---------------- 5. report ----------------
sym_f = open(os.path.join(LOG,'p0_symbols.txt'),'w')
hdr = "%-22s %-9s %-9s %6s %6s %6s %9s %8s %8s %7s" % (
    'instantiation','start','end','VGPR','AGPR','SGPR','scratch','spillV','spillS','maxwg')
print(hdr); sym_f.write(hdr+"\n")
print('-'*len(hdr)); sym_f.write('-'*len(hdr)+"\n")
for nm, a, b in funcs:
    m = meta.get(nm, {})
    d = adir.get(nm, {})
    row = "%-22s %-9d %-9d %6s %6s %6s %9s %8s %8s %7s" % (
        tag(nm), a, b, m.get('vgpr','?'), m.get('agpr','?'), m.get('sgpr','?'),
        m.get('scratch','?'), m.get('spill_v','-'), m.get('spill_s','-'), m.get('maxwg','?'))
    print(row); sym_f.write(row+"\n")
    sym_f.write("    mangled: %s\n" % nm)
    sym_f.write("    amdhsa: next_free_vgpr=%s next_free_sgpr=%s accum_offset=%s "
                "private_segment_fixed_size=%s group_segment_fixed_size=%s\n" % (
                d.get('next_free_vgpr','?'), d.get('next_free_sgpr','?'),
                d.get('accum_offset','?'), d.get('private_segment_fixed_size','?'),
                d.get('group_segment_fixed_size','?')))
sym_f.close()

# scratch sites across the whole file, attributed to function + loop nest
scr_f = open(os.path.join(LOG,'p0_scratch.txt'),'w')
loop_f = open(os.path.join(LOG,'p0_loops.txt'),'w')
print()
print("################ LOOP NEST + SCRATCH ATTRIBUTION ################")
for nm, a, b in funcs:
    t = tag(nm)
    lps = loops_of(a, b)
    ops = instrs(a, b)
    scratch = [(i,mn,st) for (i,mn,st) in ops if mn.startswith('scratch_')]
    mfma    = [(i,mn,st) for (i,mn,st) in ops if mn.startswith('v_mfma')]
    loop_f.write("\n===== %s  %s  lines %d..%d  (%d instrs, %d loops, %d mfma, %d scratch)\n"
                 % (t, nm, a, b, len(ops), len(lps), len(mfma), len(scratch)))
    for (ls, le, bt, lbl) in lps:
        lops = instrs(ls, le)
        nm_c = collections.Counter(x[1] for x in lops)
        nmfma = sum(v for k,v in nm_c.items() if k.startswith('v_mfma'))
        inner = [1 for (s2,e2,_,_) in lps if s2>=ls and e2<=le and (s2,e2)!=(ls,le)]
        loop_f.write("   loop %6d..%-6d len=%-5d  %-16s -> %-14s  mfma=%-4d nested=%d\n"
                     % (ls, le, le-ls, bt, lbl, nmfma, len(inner)))
    if scratch or mfma:
        print()
        print("=== %s  lines %d..%d : %d loops, %d mfma, %d scratch" % (t, a, b, len(lps), len(mfma), len(scratch)))
        for (ls, le, bt, lbl) in lps:
            lops = instrs(ls, le)
            nmfma = sum(1 for x in lops if x[1].startswith('v_mfma'))
            nscr  = sum(1 for x in lops if x[1].startswith('scratch_'))
            if nmfma or nscr:
                print("    LOOP %d..%d (len %d, %s) mfma=%d scratch=%d"
                      % (ls, le, le-ls, bt, nmfma, nscr))
    for (i, mn, st) in scratch:
        encl = [(ls,le) for (ls,le,_,_) in lps if ls <= i <= le]
        encl.sort(key=lambda p: p[1]-p[0])
        desc = ("INSIDE loops " + ",".join("%d..%d"%p for p in encl)) if encl else "NOT IN ANY LOOP (straight-line)"
        line = "%-22s L%-7d %-60s   %s" % (t, i, st, desc)
        scr_f.write(line+"\n")
        print("    SCRATCH " + line)
scr_f.close(); loop_f.close()

# ---------------- 6. deep dive on 256/256/32 ----------------
inv = open(os.path.join(LOG,'p0_mainloop_inventory.txt'),'w')
for nm, a, b in funcs:
    t = tag(nm)
    if not t.startswith('<256,256,32'):
        continue
    lps = loops_of(a, b)
    # innermost loop with the most mfma
    cand = []
    for (ls, le, bt, lbl) in lps:
        lops = instrs(ls, le)
        nmfma = sum(1 for x in lops if x[1].startswith('v_mfma'))
        if nmfma: cand.append((le-ls, ls, le, bt, nmfma, len(lops)))
    cand.sort()
    inv.write("\n" + "="*100 + "\n")
    inv.write("SYMBOL %s  %s\n  function lines %d..%d\n" % (t, nm, a, b))
    inv.write("  loops containing mfma (sorted by size, innermost first):\n")
    for (sz, ls, le, bt, nmfma, ni) in cand:
        inv.write("    %6d..%-6d len=%-5d %-16s mfma=%-4d instrs=%d\n" % (ls, le, sz, bt, nmfma, ni))
    if not cand: 
        inv.write("  NO LOOP CONTAINS MFMA -> mainloop fully unrolled/flattened?\n")
        continue
    sz, ls, le, bt, nmfma, ni = cand[0]
    ops = instrs(ls, le)
    c = collections.Counter(x[1] for x in ops)
    inv.write("\n  ---- INNERMOST MFMA LOOP: lines %d..%d (body %d lines, %d instrs) ----\n" % (ls, le, le-ls, ni))
    inv.write("  full mnemonic histogram:\n")
    for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        inv.write("     %6d  %s\n" % (v, k))
    inv.write("\n  waitcnt detail (line: full operand):\n")
    for (i, mn, st) in ops:
        if mn == 's_waitcnt':
            inv.write("     L%-7d %s\n" % (i, st))
    inv.write("\n  barriers / setprio / sched:\n")
    for (i, mn, st) in ops:
        if mn in ('s_barrier','s_setprio','s_nop') or 'sched' in mn:
            inv.write("     L%-7d %s\n" % (i, st))
    inv.write("\n  PROGRAM ORDER (run-length encoded, sync-relevant only):\n")
    for (k, n, i0, i1) in seq(ops):
        inv.write("     x%-5d %-34s  L%d..L%d\n" % (n, k, i0, i1))
    # dump the loop body + context
    fn = os.path.join(ISA, "mainloop_%s.s" % t.replace('<','').replace('>','').replace(',','_'))
    with open(fn, 'w') as g:
        lo, hi = max(1, ls-80), min(len(lines), le+80)
        g.write("; extracted from %s\n; symbol %s\n; %s\n" % (s_path, nm, t))
        g.write("; innermost mfma loop = lines %d..%d ; dumped %d..%d with context\n" % (ls, le, lo, hi))
        for i in range(lo, hi+1):
            mark = '>>' if ls <= i <= le else '  '
            g.write("%s %7d| %s\n" % (mark, i, lines[i-1]))
    inv.write("\n  body dump -> %s\n" % fn)
inv.close()
print()
print("wrote: logs/p0_symbols.txt logs/p0_loops.txt logs/p0_scratch.txt logs/p0_mainloop_inventory.txt")
PYEOF

if command -v python3 >/dev/null 2>&1; then
  echo "using host python3: $(python3 -V 2>&1)"
  python3 "$EXP/analyze_isa.py" "$S" "$EXP" 2>&1 | tee "$EXP/logs/p0_10_analyze.stdout.txt"
else
  echo "using container python3"
  docker exec dhk-gemmrs python3 "$EXP/analyze_isa.py" "$S" "$EXP" 2>&1 | tee "$EXP/logs/p0_10_analyze.stdout.txt"
fi
echo "=============== DONE ==============="