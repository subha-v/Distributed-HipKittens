#!/usr/bin/env bash
# Look for gemm-rs benchmark numbers already recorded ON THIS NODE by earlier
# campaigns, and for the reference (GEMM+RCCL) arm from the first attempt.
set -uo pipefail

echo "===== exp027 (stock gemm-rs BENCHMARK) recorded output ====="
for d in /home/subvadla/ddt-exp027-o1-stock-gemm-rs-benchmark-ccf585d6 \
         /home/subvadla/ddt-exp027-o1-stock-gemm-rs-benchmark-ccf585d6-lead; do
  echo "--- $d ---"
  [ -d "$d" ] || { echo "absent"; continue; }
  find "$d" -name 'index.json' 2>/dev/null | head -3
  python3 - "$d" <<'PY'
import json, sys, glob, os, base64
root = sys.argv[1]
for p in glob.glob(os.path.join(root, "runtime", "*", "index.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print("  unreadable", p, e); continue
    ex = d.get("execution", {})
    print("  argv:", ex.get("argv"))
    print("  elapsed_s:", ex.get("elapsed_s"))
    caps = ex.get("captures", {})
    pc = caps.get("popcorn", {})
    txt = pc.get("utf8_replace", "")
    if txt:
        print("  --- popcorn ---")
        for line in txt.splitlines():
            print("   ", line)
PY
done

echo
echo "===== reference (GEMM+RCCL) arm from the first attempt ====="
D=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight/compbench/reference
ls -la "$D" 2>/dev/null | head
for f in "$D"/benchmark.popcorn.txt "$D"/*.popcorn.txt; do
  [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }
done
echo "--- stderr tail if it failed ---"
tail -12 "$D"/benchmark.stderr.txt 2>/dev/null

echo
echo "===== any other recorded gemm-rs benchmark results in amd-master ====="
ls -la /home/subvadla/amd-master/auto-gpu-kernel/k1_comm_overlap/harness/results/ 2>/dev/null | head -20
echo "--- solprobe (what SOL was measured against) ---"
head -c 1500 /home/subvadla/amd-master/auto-gpu-kernel/k1_comm_overlap/harness/results/solprobe_gemm-rs.json 2>/dev/null
