#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
python3 - "$E22/b0_kernel_map.json" "$E22/events_b0_reference.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for section in ("classified", "unclassified"):
    print(f"===== {section}: {len(m[section])} names =====")
    for name, e in sorted(m[section].items(), key=lambda kv: -kv[1]["ns"]):
        print(f"{e['ns']/1e3:10.1f}us x{e['count']:<5} {str(e.get('resource')):>8} "
              f"rule={str(e.get('matched_rule'))[:16]:<18} {name[:62]}")
ev = json.load(open(sys.argv[2]))
print(f"===== chosen epoch: {ev['epoch_ns']/1e3:.1f} us, "
      f"{len(ev['intervals'])} intervals, usable={ev['epoch_count_usable']} "
      f"of {ev['epoch_count_traced']} =====")
for i in ev["intervals"]:
    print(f"  {i['begin_ns']/1e3:8.1f} -> {i['end_ns']/1e3:8.1f} us  "
          f"({(i['end_ns']-i['begin_ns'])/1e3:7.1f})  {i['resource']:>5}  {i['name'][:56]}")
d = sorted(ev["epoch_durations_ns"])
print(f"epoch durations us: min={d[0]/1e3:.1f} med={d[len(d)//2]/1e3:.1f} max={d[-1]/1e3:.1f}")
PY
echo "===== lease / queue ====="
bash /home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/tools/gpu_lease.sh status 2>&1 | head -4
