#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
python3 - "$E22/timeline_bins.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
print("columns:", rows[0].keys())
by = collections.defaultdict(list)
for r in rows: by[r["arm"]].append(r)
for arm, rs in by.items():
    f = lambda k: [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    mf, hb, xg = f("mfma_frac"), f("hbm_gbs"), f("xgmi_gbs")
    dur = float(rs[-1]["t_us_end"]) - float(rs[0]["t_us_start"])
    print(f"\n== {arm}: {len(rs)} bins, {dur:.0f} us, "
          f"denominator {rs[0].get('mfma_denominator')}")
    for name, v in (("mfma_frac", mf), ("hbm_gbs", hb), ("xgmi_gbs", xg)):
        if v: print(f"   {name:<10} peak={max(v):8.2f}  mean={sum(v)/len(v):8.2f}"
                    f"  bins>0={sum(1 for x in v if x>0):3d}")
    # overlap: bins where the MFMA proxy and the xGMI strip are both live
    both = sum(1 for r in rs
               if float(r["mfma_frac"]) > 0.05 and float(r["xgmi_gbs"]) > 1.0)
    print(f"   bins with MFMA>5% AND xGMI>1 GB/s: {both} of {len(rs)} "
          f"({both/len(rs):.0%})  <-- the overlap claim")
PY
