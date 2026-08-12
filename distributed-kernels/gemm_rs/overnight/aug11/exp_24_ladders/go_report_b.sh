#!/usr/bin/env bash
# Final aggregation: instrument A (6 shapes) + instrument B (1 rotation), then the
# report, then the A-vs-B ordering cross-check that is instrument B's whole purpose.
set -uo pipefail
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders

echo "===== aggregate strict: expect-a 6, expect-b 1 ====="
docker exec dhk-gemmrs bash -lc \
  "cd $D && python3 ladders.py --root $D --out $D/ladders.json --ladder-dir ladder --expect-a 6 --expect-b 1" 2>&1 | tail -8

echo
echo "===== instrument B, per arm, per shape (us) ====="
docker exec dhk-gemmrs python3 - <<'PY'
import json, math
D="/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_24_ladders"
doc=json.load(open(f"{D}/ladders.json"))
cc=doc.get("evaluator_crosscheck",{})
LB=doc["config"]["shapes"]

def geo(v):
    v=[x for x in v if x]
    return math.exp(sum(map(math.log,v))/len(v)) if v else None

rows={}
for rot,arms in sorted(cc.get("rotations",{}).items()):
    for name,p in sorted(arms.items()):
        tag="WARM(throwaway)" if p.get("is_warm") else ("headline" if p.get("is_headline") else "?")
        best=[s.get("best_us") for s in p.get("per_shape",[])]
        rows[name]=(tag,best,p.get("geomean_us"),p.get("check"))

hdr=f"{'arm/pass':<26}{'chk':>6}" + "".join(f"{l.split('x')[0]:>10}" for l in LB) + f"{'geomean':>11}"
print(hdr); print("-"*len(hdr))
for name,(tag,best,gm,chk) in rows.items():
    line=f"{name:<26}{str(chk):>6}"
    for b in best: line += f"{b:>10.1f}" if b else f"{'--':>10}"
    line += f"{gm:>11.1f}" if gm else f"{'--':>11}"
    print(line + ("   " + tag if tag!="headline" else ""))

# The point of instrument B: does the one-process-per-rank topology change the
# ORDER of the arms, or only the absolutes?
print()
def gm_of(sub):
    for k,(t,b,g,c) in rows.items():
        if k.endswith(sub): return g
    return None
o,r,k = gm_of("ours/benchmark"), gm_of("reference/benchmark"), gm_of("rank1/bench")
A=doc["protocols"]["graded"]["arms"]
ao,ar,ak = A["ours"]["geomean_us"], A["reference"]["geomean_us"], A["rank1"]["geomean_us"]
print("=== A vs B: absolutes and ORDERING ===")
print(f"{'arm':<12}{'A graded':>11}{'B evaluator':>13}{'B/A':>8}")
for nm,a,b in (("ours",ao,o),("reference",ar,r),("rank1",ak,k)):
    if a and b: print(f"{nm:<12}{a:>11.1f}{b:>13.1f}{b/a:>8.2f}x")
if o and k and r:
    print(f"\nratios: A  ours/rank1 {ao/ak:.4f}  ours/reference {ao/ar:.4f}")
    print(f"        B  ours/rank1 {o/k:.4f}  ours/reference {o/r:.4f}")
    oa=sorted([("ours",ao),("reference",ar),("rank1",ak)],key=lambda x:x[1])
    ob=sorted([("ours",o),("reference",r),("rank1",k)],key=lambda x:x[1])
    print(f"        A order (fastest first): {[x[0] for x in oa]}")
    print(f"        B order (fastest first): {[x[0] for x in ob]}")
    print(f"        ORDERING {'AGREES' if [x[0] for x in oa]==[x[0] for x in ob] else 'DISAGREES'}")
print(f"\nparse refusals: {cc.get('parse_errors')}")
print(f"test-mode files skipped: {len(cc.get('skipped_test_mode',[]))}")
PY
