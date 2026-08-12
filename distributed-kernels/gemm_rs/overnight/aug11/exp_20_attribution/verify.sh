#!/usr/bin/env bash
# Final check: deliverables present, parseable, and internally consistent.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
OUT=$ON/aug11/exp_20_attribution

echo "===== deliverables ====="
ls -la "$OUT"/ablation.json "$OUT"/counters.json "$OUT"/result.md \
       "$OUT"/m7_results.json 2>&1

echo
echo "===== JSON validity + required keys ====="
python3 - "$OUT" << 'PY'
import json, sys, os
out = sys.argv[1]
a = json.load(open(os.path.join(out, "ablation.json")))
c = json.load(open(os.path.join(out, "counters.json")))
need = {"full_us", "stages_us", "stage_share", "floor_us", "geometry",
        "x_sol", "m7"}
print(f"  ablation.json: {len(a['shapes'])} shapes, ranking entries "
      f"{len(a['ranking'])}, gate passed={a['freshness_gate']['passed']}")
for r in a["shapes"]:
    missing = need - set(r)
    st = set(r["stages_us"])
    assert st == {"gemm", "egress", "reduce", "sync", "release"}, st
    print(f"    shape {r['shape_index']:>1} {r['shape']:<16} "
          f"full={r['full_us']:>7.1f} "
          f"w*ki={r['geometry']['waves_x_k_iters']:>4} "
          f"rgrp={r['geometry']['effective_release_group']} "
          f"bound={r['m7'].get('bound','-'):<6} "
          f"f/M7={r['m7']['full_vs_m7_mean']:.3f}"
          + (f"  MISSING {missing}" if missing else ""))
print(f"  counters.json: {len(c['cells'])} cells, {len(c['shapes'])} rollups")
for r in c["shapes"]:
    print(f"    shape {r['shape_index']:>1} {r['shape']:<16} "
          f"amp={r['amplification']:.4f} 64B={100*r['wrreq_64b_share']:.1f}% "
          f"lat={r.get('ea_write_latency_cycles','?')} "
          f"usable={r['usable']} groups={''.join(r['groups'])}")
bad = [t for t, v in c["cells"].items()
       if "errors=none" not in v.get("prof_line", "")]
print(f"  cells with a nonzero error bit: {bad if bad else 'NONE'}")
print("  OK")
PY

echo
echo "===== result.md ====="
wc -l "$OUT/result.md"
head -14 "$OUT/result.md"
