#!/usr/bin/env bash
# P0 step 5: where do the 126 counted vmcnt(N) waits live, and what produced the ISA?
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
EXP=$ON/experiments/exp_03_mainloop
S=$ON/build/isa/gemm_rs_mi300x-hip-amdgcn-amd-amdhsa-gfx942.s

echo "############ provenance: isa/build.log ############"
cat "$ON/build/isa/build.log" 2>/dev/null | head -30
echo "(build.log above; note it is the --save-temps compile driven by tools/m2_isa.sh)"

echo
echo "############ counted vmcnt(N>0): line numbers + which symbol/loop ############"
grep -nE 'vmcnt\([1-9]' "$S" | awk -F: '{print $1}' > /tmp/cnt.txt
python3 - "$S" /tmp/cnt.txt <<'PYEOF'
import re,sys,collections
lines=open(sys.argv[1],errors='replace').read().splitlines()
nums=[int(x) for x in open(sys.argv[2])]
re_fs=re.compile(r'^(_Z\S+):\s*(;.*)?$'); re_fe=re.compile(r'^\.Lfunc_end')
funcs=[];cur=None
for i,L in enumerate(lines,1):
    m=re_fs.match(L)
    if m: cur=(m.group(1),i); continue
    if re_fe.match(L) and cur: funcs.append((cur[0],cur[1],i)); cur=None
def tag(n):
    nu=re.findall(r'Li(\d+)E',n); tl=re.search(r'Lb(\d)E',n)
    return "<%s,%s,%s,%s>"%(nu[0],nu[1],nu[2],'true' if tl and tl.group(1)=='1' else 'false') if len(nu)>=3 else n
c=collections.Counter(); ranges=collections.defaultdict(list)
for n in nums:
    for nm,a,b in funcs:
        if a<=n<=b:
            c[tag(nm)]+=1; ranges[tag(nm)].append(n); break
for k in sorted(c, key=lambda x:-c[x]):
    r=ranges[k]
    print("  %-22s %3d counted vmcnt, lines %d..%d"%(k,c[k],min(r),max(r)))
# is any inside the k-loop of the 256/256/32 pair?
KL={'<256,256,32,false>':(10957,11260),'<256,256,32,true>':(16012,16438)}
for k,(a,b) in KL.items():
    inside=[n for n in ranges.get(k,[]) if a<=n<=b]
    print("  %-22s counted vmcnt INSIDE k-loop %d..%d : %d  %s"%(k,a,b,len(inside),inside[:10]))
print()
print("  context of the first 3 counted vmcnt (proof the compiler emits them on gfx942):")
for n in nums[:3]:
    for i in range(max(1,n-4),min(len(lines),n+2)+1):
        print("    %s %7d| %s"%('>>' if i==n else '  ',i,lines[i-1]))
    print()
PYEOF
echo "=============== DONE ==============="