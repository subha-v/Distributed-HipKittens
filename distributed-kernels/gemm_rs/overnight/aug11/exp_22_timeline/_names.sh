#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
python3 - "$E22/b0_kernel_map.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for section in ("classified", "unclassified"):
    entries = doc[section]
    print(f"===== {section}: {len(entries)} names =====")
    for name, e in sorted(entries.items(), key=lambda kv: -kv[1]["ns"]):
        print(f"{e['ns']/1e3:11.1f} us  x{e['count']:<6} {str(e.get('resource')):>5}  "
              f"rule={str(e.get('matched_rule'))[:20]:<22} {name[:80]}")
PY
