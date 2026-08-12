#!/usr/bin/env bash
# exp_26 step 3e: .text byte-identity gate, scratch-op localization, wait-site
# detail for the phase-1 K-loop.
set -uo pipefail
SC=$HOME/overnight-scratch/e26

cat > "$SC/py/detail.py" <<'PYEOF'
import re, sys, collections
def load(path):
    insns, base = [], None
    for line in open(path, errors='replace'):
        m = re.match(r'^([0-9a-f]{16}) <([^>]+)>:', line)
        if m: base = int(m.group(1),16); continue
        m = re.match(r'^\t(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):\s*(.*)$', line.rstrip('\n'))
        if not m: continue
        tgt=None
        mt = re.search(r'<[^>]*\+0x([0-9a-f]+)>', m.group(4))
        if mt and base is not None: tgt = base + int(mt.group(1),16)
        insns.append((int(m.group(3),16), m.group(1), m.group(2), tgt))
    return insns
def cls(m):
    if m.startswith('v_mfma'): return 'MFMA'
    if m.startswith('ds_read'): return 'DSR'
    if m.startswith('ds_write'): return 'DSW'
    if m.startswith('scratch_'): return 'SCR'
    if m.startswith(('buffer_load','global_load','flat_load')): return 'VMLD'
    if m=='s_waitcnt': return 'WAIT'
    if m.startswith('s_barrier'): return 'BAR'
    return 'oth'

P1 = {'B0':(0xb050,0xc464), 'B1':(0xb050,0xc464), 'B2':(0xb098,0xc434)}
P2 = {'B0':(0x10b40,0x11c34),'B1':(0x10b40,0x11c34),'B2':(0x10b80,0x11c74)}

for tag in ('B0','B1','B2'):
    ins = load(f'{tag}.isa')
    print(f'===== {tag} =====')
    # 1. scratch localization
    scr = [(a,m) for a,m,_,_ in ins if m.startswith('scratch_')]
    print(f'  scratch_ops={len(scr)}')
    buckets = collections.Counter()
    for a,m in scr:
        if   P1[tag][0] <= a <= P1[tag][1]: buckets['INSIDE_P1_KLOOP'] += 1
        elif P2[tag][0] <= a <= P2[tag][1]: buckets['INSIDE_P2_KLOOP'] += 1
        elif a < P1[tag][0]:                buckets['before_M6'] += 1
        elif a < P2[tag][0]:                buckets['M6_tail..M7_head'] += 1
        else:                               buckets['after_P2_KLOOP (M7 epi/M8/M9)'] += 1
    for k,v in sorted(buckets.items()): print(f'    {k}: {v}')
    lo = min(a for a,_ in scr) if scr else 0
    hi = max(a for a,_ in scr) if scr else 0
    print(f'    scratch addr range 0x{lo:x}..0x{hi:x}')
    print(f'    p1_kloop=0x{P1[tag][0]:x}..0x{P1[tag][1]:x}  p2_kloop=0x{P2[tag][0]:x}..0x{P2[tag][1]:x}')
    # 2. wait-site detail inside the phase-1 K-loop
    b = [x for x in ins if P1[tag][0] <= x[0] <= P1[tag][1]]
    nm=nv=0; rows=[]
    for a,m,o,_ in b:
        c = cls(m)
        if c=='MFMA': nm+=1
        if c=='VMLD': nv+=1
        if c=='WAIT': rows.append((a,o,nm,nv))
        if c=='BAR':  rows.append((a,'--BARRIER--',nm,nv))
    print('  phase-1 K-loop wait/barrier timeline  (mfma_so_far, vmld_issued_so_far)')
    for a,o,nm_,nv_ in rows: print(f'    0x{a:x}  {o:<24s} mfma={nm_:<3d} vmld={nv_:<3d}')
    print(f'    totals: mfma={nm} vmld={nv}')
PYEOF

docker exec subha_k1 bash -lc '
SC=/home/subvadla/overnight-scratch/e26
echo "=== .text SECTION BYTE IDENTITY (the real B1==B0 gate) ==="
for v in B0 B1 B2; do
  /opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=.text $SC/out/$v.gfx950.elf $SC/out/$v.text.bin 2>/dev/null
done
sha256sum $SC/out/*.text.bin
ls -l $SC/out/*.text.bin
echo "=== WHERE DO B0/B1 ELFs DIFFER AT ALL? ==="
cmp -l $SC/out/B0.gfx950.elf $SC/out/B1.gfx950.elf | wc -l
cmp -l $SC/out/B0.gfx950.elf $SC/out/B1.gfx950.elf | head -5
echo "--- differing byte offsets (decimal, first/last) ---"
cmp -l $SC/out/B0.gfx950.elf $SC/out/B1.gfx950.elf | awk "NR==1{print \"first=\" \$1} END{print \"last=\" \$1}"
echo "--- strings containing tu/b0 or tu/b1 ---"
strings $SC/out/B0.gfx950.elf | grep -c "tu/b0"
strings $SC/out/B1.gfx950.elf | grep -c "tu/b1"
echo "=== NOTES DIFF B0 vs B1 ==="
diff <(/opt/rocm/llvm/bin/llvm-readelf --notes $SC/out/B0.gfx950.elf) <(/opt/rocm/llvm/bin/llvm-readelf --notes $SC/out/B1.gfx950.elf) | head -20
echo "(end notes diff)"
echo
cd $SC/out && python3 ../py/detail.py
'
