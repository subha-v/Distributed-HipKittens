#!/usr/bin/env bash
# Hand-run the check SIInsertWaitcnts cannot do here, for BOTH counters.
#
# Every memory op in include/cdna3/** is inside an asm volatile, so the
# compiler's waitcnt pass never sees the event and inserts no use-wait of its
# own; the only waits in the mainloop are the ones the source asks for. And the
# asm's output operand makes the compiler believe the destination register is
# live immediately, so it will happily schedule a consumer between the load and
# its wait. That is a silent wrong-answer bug, so check it mechanically:
#
#   lgkmcnt: destinations of asm ds_read since the last lgkmcnt drain
#   vmcnt:   destinations of asm global/buffer/flat load since the last vmcnt drain
#
# and report any instruction that READS one of those registers, plus any
# s_barrier reached with asm LDS reads still in flight.
#
# Only loads issued INSIDE asm volatile are tracked (the compiler manages its
# own correctly, and tracking those produces only false positives), but ALL
# s_waitcnt instructions are honoured as drains, whoever emitted them. Counted
# waits are handled properly: vmcnt(N) retires all but the newest N loads.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
S=${1:-$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s}

python3 - "$S" <<'PY'
import re, sys
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

def regs(text):
    out = set()
    for m in re.finditer(r'\b([va])\[(\d+):(\d+)\]', text):
        k, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        for r in range(lo, hi + 1): out.add(f"{k}{r}")
    for m in re.finditer(r'(?<![\w\[:])([va])(\d+)\b', text):
        out.add(f"{m.group(1)}{m.group(2)}")
    return out

VM_LOAD  = ('global_load', 'buffer_load', 'flat_load', 'scratch_load')
LGKM_LD  = ('ds_read',)

total_haz = 0
for nm, a, b in funcs:
    t = tag(nm)
    print("=" * 96)
    print(f"SYMBOL {t}   lines {a}..{b}")
    vmq, lgq = [], []        # queues of (line, mnemonic, {regs}), oldest first
    hazards = []
    inasm = False
    for i in range(a, b + 1):
        raw = lines[i - 1]
        if ';;#ASMSTART' in raw: inasm = True;  continue
        if ';;#ASMEND'   in raw: inasm = False; continue
        st = raw.split(';')[0].strip()
        if not st or st.startswith('.') or st.endswith(':'): continue
        mn = st.split()[0]

        if mn.startswith('s_waitcnt'):
            mv = re.search(r'vmcnt\((\d+)\)', st)
            ml = re.search(r'lgkmcnt\((\d+)\)', st)
            if mv is not None: vmq = vmq[len(vmq) - int(mv.group(1)):] if int(mv.group(1)) else []
            if ml is not None: lgq = lgq[len(lgq) - int(ml.group(1)):] if int(ml.group(1)) else []
            continue
        if mn == 's_barrier':
            if lgq:
                hazards.append((i, 's_barrier, asm LDS reads still in flight',
                                [(r, e[0]) for e in lgq for r in sorted(e[2])][:4]))
            continue

        ops = st.split(None, 1)[1] if ' ' in st else ''
        if inasm and mn.startswith(LGKM_LD):
            lgq.append((i, mn, regs(ops.split(',')[0]))); continue
        if inasm and mn.startswith(VM_LOAD):
            vmq.append((i, mn, regs(ops.split(',')[0]))); continue

        used = regs(ops)
        for label, q in (('lgkm', lgq), ('vm', vmq)):
            hit = [(r, e[0]) for e in q for r in sorted(used & e[2])]
            if hit:
                hazards.append((i, f"{mn} [{label}]", hit[:4]))
                for e in q: e[2].difference_update(used)
    if not hazards:
        print("  clean: nothing consumed before its load was drained")
    else:
        total_haz += len(hazards)
        print(f"  {len(hazards)} HAZARD(S):")
        for (i, mn, det) in hazards[:25]:
            print(f"    L{i:<7} {mn:<34} reads " +
                  ", ".join(f"{r}<-L{ln}" for r, ln in det))
        if len(hazards) > 25: print(f"    ... {len(hazards)-25} more")
print("=" * 96)
print(f"TOTAL HAZARDS ACROSS ALL SYMBOLS: {total_haz}")
PY
