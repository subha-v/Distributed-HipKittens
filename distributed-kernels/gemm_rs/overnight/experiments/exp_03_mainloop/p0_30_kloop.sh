#!/usr/bin/env bash
# P0 step 3: basic-block-accurate k-loop inventory for the 256/256/32 pair.
# READ-ONLY. No GPU, no build.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s

cat > "$EXP/kloop_isa.py" <<'PYEOF'
import re, sys, os, collections
s_path, outdir = sys.argv[1], sys.argv[2]
lines = open(s_path, errors='replace').read().splitlines()
LOG, ISA = os.path.join(outdir,'logs'), os.path.join(outdir,'isa')

# ---- functions ----
re_fstart = re.compile(r'^(_Z\S+):\s*(;.*)?$')
re_fend   = re.compile(r'^\.Lfunc_end(\d+):')
funcs, cur = [], None
for i,L in enumerate(lines,1):
    m = re_fstart.match(L)
    if m: cur=(m.group(1),i); continue
    if re_fend.match(L) and cur: funcs.append([cur[0],cur[1],i]); cur=None
def tag(n):
    nu=re.findall(r'Li(\d+)E',n); tl=re.search(r'Lb(\d)E',n)
    return "<%s,%s,%s,%s>"%(nu[0],nu[1],nu[2],'true' if tl and tl.group(1)=='1' else 'false') if len(nu)>=3 else n

# ---- basic blocks + loop annotations ----
re_blk = re.compile(r'^(\.LBB[\w.$]+):|^; (%bb\.[\w.$]+):')
re_inloop = re.compile(r'in Loop: Header=(BB[\w.$]+) Depth=(\d+)')
re_hdr    = re.compile(r'=>\s*This (Loop|Inner Loop) Header: Depth=(\d+)')

def blocks_of(a,b):
    out=[]
    for i in range(a,b+1):
        m=re_blk.match(lines[i-1])
        if m:
            name=m.group(1) or m.group(2)
            ann=lines[i-1]
            if i<len(lines): ann+=" || "+lines[i]
            if i+1<len(lines): ann+=" || "+lines[i+1]
            h=re_inloop.search(ann); hd=re_hdr.search(ann)
            out.append(dict(name=name,start=i,hdr=(h.group(1) if h else None),
                            depth=int(h.group(2)) if h else (int(hd.group(2)) if hd else 0),
                            is_hdr=bool(hd), ann=lines[i-1].strip()))
    for k in range(len(out)):
        out[k]['end'] = (out[k+1]['start']-1) if k+1<len(out) else b
    return out

def instrs(a,b):
    o=[]
    inasm=False
    for i in range(a,b+1):
        raw=lines[i-1]
        st=raw.split(';')[0].strip()
        if ';;#ASMSTART' in raw: inasm=True;  continue
        if ';;#ASMEND'   in raw: inasm=False; continue
        if not st or st.startswith('.') or st.endswith(':'): continue
        o.append((i,st.split()[0],st,inasm))
    return o

SYNC=('global_load','global_store','global_atomic','buffer_load','buffer_store','buffer_wbl2',
      'buffer_inv','buffer_atomic','flat_load','flat_store','flat_atomic','ds_read','ds_write',
      'ds_bpermute','ds_add','s_waitcnt','s_barrier','v_mfma','s_setprio','scratch_store',
      'scratch_load','s_sleep','v_accvgpr')
def issync(m): return any(m.startswith(p) for p in SYNC)

out=open(os.path.join(LOG,'p0_kloop_inventory.txt'),'w')
def P(s=''):
    print(s); out.write(s+"\n")

for nm,a,b in funcs:
    t=tag(nm)
    if not t.startswith('<256,256,32'): continue
    blks=blocks_of(a,b)
    # group by loop header
    byhdr=collections.defaultdict(list)
    for bl in blks:
        if bl['hdr']: byhdr[(bl['hdr'],bl['depth'])].append(bl)
    # a header block itself is annotated "=>This Loop Header"; attach it
    hdrblk={}
    for bl in blks:
        if bl['is_hdr']: hdrblk[bl['name'].replace('.L','')]=bl
    P(); P('#'*110)
    P("SYMBOL %s   %s   lines %d..%d"%(t,nm,a,b))
    # find the deepest loop that contains v_mfma
    best=None
    for (h,d),bs in sorted(byhdr.items(), key=lambda kv:-kv[0][1]):
        allb=list(bs)
        hb=hdrblk.get(h)
        if hb and hb not in allb: allb.append(hb)
        allb.sort(key=lambda x:x['start'])
        nmf=sum(1 for bl in allb for x in instrs(bl['start'],bl['end']) if x[1].startswith('v_mfma'))
        P("  loop Header=%-10s Depth=%d  blocks=%-3d layout %d..%d  mfma=%d"%(
            h,d,len(allb),allb[0]['start'],allb[-1]['end'],nmf))
        if nmf and (best is None or d>best[1]): best=(h,d,allb)
    if not best: P("  NO MFMA LOOP FOUND"); continue
    h,d,allb=best
    P(); P("  ==== DEEPEST MFMA LOOP: Header=%s Depth=%d ===="%(h,d))
    # execution order: header block first, then remaining layout order, latch(es) as they fall
    hb=[x for x in allb if x['is_hdr'] or x['name'].replace('.L','')==h]
    order=hb+[x for x in allb if x not in hb]
    P("  blocks (layout order):")
    for bl in allb:
        term=[x[2] for x in instrs(bl['start'],bl['end'])
              if x[1].startswith('s_cbranch') or x[1]=='s_branch' or x[1]=='s_endpgm']
        P("    %-14s %6d..%-6d (%4d lines) role=%-8s term=%s  %s"%(
            bl['name'],bl['start'],bl['end'],bl['end']-bl['start']+1,
            'HEADER' if bl['is_hdr'] or bl['name'].replace('.L','')==h else 'body/latch',
            (term[-1] if term else 'fallthrough'), bl['ann'][:44]))
    P()
    total=collections.Counter()
    for bl in allb:
        ops=instrs(bl['start'],bl['end'])
        c=collections.Counter(x[1] for x in ops)
        total.update(c)
        P("  ---- BLOCK %s  %d..%d  (%d instrs) ----"%(bl['name'],bl['start'],bl['end'],len(ops)))
        keys=sorted(c.items(), key=lambda kv:(-kv[1],kv[0]))
        P("     hist: " + "  ".join("%s=%d"%(k,v) for k,v in keys if issync(k) or k.startswith('s_')))
        P("     sync-relevant, PROGRAM ORDER (A=inline asm, C=compiler):")
        run=[]
        for (i,mn,st,ia) in ops:
            if not issync(mn): continue
            key=(st if mn=='s_waitcnt' else mn, 'A' if ia else 'C')
            if run and run[-1][0]==key: run[-1][1]+=1; run[-1][3]=i
            else: run.append([key,1,i,i])
        for (key,n,i0,i1) in run:
            P("        [%s] x%-4d %-42s L%d..L%d"%(key[1],n,key[0],i0,i1))
        P()
    P("  ==== TOTAL over all blocks of the k-loop ====")
    for k,v in sorted(total.items(), key=lambda kv:(-kv[1],kv[0])):
        if issync(k) or k.startswith('s_b') or k.startswith('s_w'):
            P("     %6d  %s"%(v,k))
    P("     (all mnemonics, top 25)")
    for k,v in sorted(total.items(), key=lambda kv:(-kv[1],kv[0]))[:25]:
        P("     %6d  %s"%(v,k))
    # dump raw text of the loop
    fn=os.path.join(ISA,"kloop_%s.s"%t.strip('<>').replace(',','_'))
    with open(fn,'w') as g:
        g.write("; %s  symbol %s\n; k-loop blocks:\n"%(t,nm))
        for bl in allb: g.write(";   %s %d..%d\n"%(bl['name'],bl['start'],bl['end']))
        lo,hi=allb[0]['start']-40, allb[-1]['end']+40
        for i in range(max(1,lo),min(len(lines),hi)+1):
            inb=any(bl['start']<=i<=bl['end'] for bl in allb)
            g.write("%s %7d| %s\n"%('>>' if inb else '  ',i,lines[i-1]))
    P("  raw dump -> %s"%fn)
out.close()
PYEOF

python3 "$EXP/kloop_isa.py" "$S" "$EXP" 2>&1 | tee "$EXP/logs/p0_30_kloop.stdout.txt"

echo
echo "############ whole-file scans for the instructions Q2 asks about ############"
for pat in 'buffer_load' 'global_load_lds' 's_setprio' 'sched_barrier' 'sched_group_barrier' 'v_accvgpr' 'v_mfma_f32_32x32' 'v_mfma_f32_16x16x16bf16_1k' 'ds_read2' 'ds_read_b128' 'global_load_dwordx4' 'buffer_wbl2' 'buffer_inv' 's_sleep'; do
  printf '%-32s total=%s\n' "$pat" "$(grep -c -- "$pat" "$S")"
done
echo "=============== DONE ==============="