#!/usr/bin/env bash
python3 - <<'EOF'
import csv, statistics as st
rows=[]
for tag in ("a11base","a11noise"):
    with open(f"/home/subvadla/overnight-scratch/screen_{tag}.csv") as fh:
        for r in csv.DictReader(fh):
            r["_tag"]=tag; rows.append(r)
print(f"total rows: {len(rows)}  statuses: {sorted({r['status'] for r in rows})}")
print(f"all csv widths ok, heads: {sorted({r['head'] for r in rows})} src_rev: {sorted({r['src_rev'] for r in rows})}")

def band(a,b):
    return abs(a-b)/((a+b)/2)*100

base=[r for r in rows if r["_tag"]=="a11base"]
r1,r2=base[0],base[1]
print("\n-- a11base runs 1 vs 2 (identical cfg) --")
for col in ("prod_us","pf6gm_us","mps_us","ratio_vs_prod","ratio_vs_pf6gm"):
    a,b=float(r1[col]),float(r2[col])
    print(f"  {col:16s} {a:9.4f} {b:9.4f}   spread {band(a,b):5.2f}%")

m12=[r for r in rows if r["cfg"]=="C=16,g=33,mode=12,flush_rows=16"]
print(f"\n-- all {len(m12)} repeats of C=16,g=33,mode=12,flush_rows=16 --")
for col in ("prod_us","pf6gm_us","mps_us","ratio_vs_prod","ratio_vs_pf6gm"):
    v=[float(r[col]) for r in m12]
    mean=st.mean(v); sd=st.stdev(v); rng=(max(v)-min(v))/mean*100
    print(f"  {col:16s} mean {mean:9.4f}  min {min(v):9.4f}  max {max(v):9.4f}  range {rng:5.2f}%  sd {sd:8.4f} ({sd/mean*100:5.2f}%)")
print("\n  mps_us sequence:", [r["mps_us"] for r in m12])
print("  hsaco sequence :", [r["mps_hsaco"] for r in m12])
by={}
for r in m12: by.setdefault(r["mps_hsaco"],[]).append(float(r["mps_us"]))
for h,v in by.items(): print(f"  hash {h}: n={len(v)} mean={st.mean(v):.1f}")
EOF
echo
echo "=== final node state ==="
pgrep -af 'torchrun|mpirun' || echo "pgrep torchrun/mpirun: EMPTY"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null || echo "no lock dir"
echo "HEAD: $(git -C ~/Distributed-HipKittens rev-parse --short HEAD)"
echo "=== artifacts ==="
ls -la ~/overnight-scratch/screen_a11*.csv ~/overnight-scratch/screen_a11*.out
ls -d ~/k0-mok-a11base/* ~/k0-mok-a11noise/* | head
exit 0
