#!/usr/bin/env bash
E22=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/aug11/exp_22_timeline
echo "===== precise mtimes ====="
stat -c '%y  %10s  %n' "$E22"/events_ours_*.json "$E22"/tick_rate.json \
  "$E22"/validation.json "$E22"/timeline_bins.csv "$E22"/parity.json 2>/dev/null
echo "===== tick_rate.json ====="
cat "$E22/tick_rate.json" 2>/dev/null
echo "===== events_ours_s5.json header ====="
python3 - "$E22/events_ours_s5.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("schema","arm","kind","shape","ticks_per_us","device_us","correctness",
          "protocol_problems","drops","module_sha256"):
    if k in d: print(f"  {k}: {json.dumps(d[k])[:220]}")
print("  plan:", json.dumps(d.get("plan", {}))[:300])
r = d.get("ranks", {})
print("  ranks:", list(r)[:3], "rank0 event count:",
      len(r.get("0", {}).get("events", [])) if r else "n/a")
PY
echo "===== smoke dir ====="; ls -l "$E22/smoke" 2>&1 | head
