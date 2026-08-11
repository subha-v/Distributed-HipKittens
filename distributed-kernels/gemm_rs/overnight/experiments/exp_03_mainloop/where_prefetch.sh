#!/usr/bin/env bash
# Locate every global_load_dwordx4 / ds_write_b64 / s_waitcnt in the
# <256,256,32,*> symbols and report the enclosing basic block WITH its loop
# annotation, so a "missing" prefetch can be attributed rather than guessed at.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
S=${1:-$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s}

python3 - "$S" <<'PY'
import re, sys, collections
lines = open(sys.argv[1], errors='replace').read().splitlines()

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

re_blk = re.compile(r'^(\.LBB[\w.$]+):|^; (%bb\.[\w.$]+):')
WANT = ('global_load_dwordx4', 'ds_write_b64', 'ds_read_b64', 's_waitcnt',
        's_barrier', 'v_mfma', 'scratch_store', 'scratch_load', 'buffer_load')

for nm, a, b in funcs:
    t = tag(nm)
    if not t.startswith('<256,256,32'): continue
    # block table with annotations
    blks = []
    for i in range(a, b + 1):
        m = re_blk.match(lines[i - 1])
        if m:
            ann = " ".join(lines[i - 1:i + 2])
            h = re.search(r'in Loop: Header=(BB[\w.$]+) Depth=(\d+)', ann)
            hd = re.search(r'This (?:Loop|Inner Loop) Header: Depth=(\d+)', ann)
            blks.append(dict(name=(m.group(1) or m.group(2)), start=i,
                             hdr=h.group(1) if h else None,
                             depth=int(h.group(2)) if h else (int(hd.group(1)) if hd else 0),
                             is_hdr=bool(hd)))
    for k in range(len(blks)):
        blks[k]['end'] = blks[k + 1]['start'] - 1 if k + 1 < len(blks) else b

    print('=' * 100)
    print(f"SYMBOL {t}  {nm}  lines {a}..{b}")
    print(f"{'block':<16}{'lines':<18}{'depth':>6}{'hdr':>12}   counts")
    for bl in blks:
        c = collections.Counter()
        for i in range(bl['start'], bl['end'] + 1):
            st = lines[i - 1].split(';')[0].strip()
            if not st: continue
            mn = st.split()[0]
            for w in WANT:
                if mn.startswith(w):
                    c[st if mn == 's_waitcnt' else w] += 1
        if not c: continue
        cs = "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print(f"{bl['name']:<16}{str(bl['start'])+'..'+str(bl['end']):<18}"
              f"{bl['depth']:>6}{str(bl['hdr'] or ('SELF' if bl['is_hdr'] else '-')):>12}   {cs}")
PY
