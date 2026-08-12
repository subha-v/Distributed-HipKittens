#!/usr/bin/env python3
"""Did the smoke run produce a usable sample file? Mechanism checks only."""
import json
import statistics
import sys

d = json.load(open(sys.argv[1]))
if d.get("error"):
    print("  ERROR:")
    print("    " + "\n    ".join(d["error"].strip().splitlines()[-6:]))
    sys.exit(1)
print(f"  shape           : {d['shape_label']}  rounds={d['rounds']}")
print(f"  kernel modules  : {d['kernel_modules']}")
print(f"  PERSHAPE per arm: {d['pershape']}")
print(f"  geometry        : {d['geometry']}")
print(f"  rgroup per arm  : {d['rgroup']}  distinct={d['arms_distinct_on_this_shape']}")
print(f"  torch.equal     : {d['torch_equal']}  all_ranks={d['all_ranks_bit_identical']}")
print(f"  ordering        : {d['ordering']}, blocks of {d['perms_per_block']}")
print(f"  first 3 orders  : {d['arm_orders'][:3]}")
for proto in ("graded", "pipelined"):
    line = []
    for arm in d["per_round"]:
        v = d["per_round"][arm][proto]
        line.append(f"{arm}={statistics.median(v):.1f}" if v else f"{arm}=EMPTY")
    print(f"  {proto:>10} round medians: {'  '.join(line)}")
ok = (d["all_ranks_bit_identical"] and d["arms_distinct_on_this_shape"]
      and all(d["per_round"][a][p] for a in d["per_round"]
              for p in ("graded", "pipelined")))
print(f"  MECHANISM: {'OK -- ready for the full campaign' if ok else 'NOT READY'}")
sys.exit(0 if ok else 1)
