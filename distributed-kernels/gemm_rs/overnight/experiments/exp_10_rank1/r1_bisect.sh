#!/usr/bin/env bash
# exp_10 -- run the stage bisect, then attribute the fault address to a region.
#   $1 shape index   $2 port
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
IRISDST=/usr/local/lib/python3.10/dist-packages
SHAPE=${1:-0}
PORT=${2:-12399}

echo "kfd_pids before launch: $(ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -c kfd)"
cp "$ON/experiments/exp_10_rank1/r1_bisect.py" "$ARM/r1_bisect.py"
LOGS=$ARM/logs_bisect_s${SHAPE}
rm -rf "$LOGS"; mkdir -p "$LOGS"
rm -f "$ARM"/ipc_handles_rank*.bin

docker exec -w "$ARM" \
  -e PATH="$ON/tools/compat/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
  -e PYTHONPATH="$ON/tools/compat:$IRISDST" \
  -e PYTHONUNBUFFERED=1 \
  -e TRITON_CACHE_DIR="$ARM/.triton" \
  -e R1_OUTDIR="$LOGS" \
  -e R1_TIMEOUT=300 \
  -e HSA_ENABLE_COREDUMP=0 \
  -e AMD_SERIALIZE_KERNEL=3 \
  dhk-gemmrs bash -lc "timeout --signal=TERM 340 setsid python3 -u r1_bisect.py $SHAPE $PORT" \
  > "$LOGS/driver.txt" 2>&1
echo "driver rc=$?"

echo
echo "===== driver summary ====="
sed -n '/driver summary/,$p' "$LOGS/driver.txt"

echo
echo "===== per-rank stage progress ====="
for r in 0 1 2 3 4 5 6 7; do
  f="$LOGS/stdout_rank$r.txt"
  echo "--- rank $r ---"
  grep -E 'STAGE|FAILED|clean exit' "$f" 2>/dev/null | sed 's/^/    /'
done

echo
echo "===== faults, with the address attributed to a region ====="
python3 - "$LOGS" <<'PYEOF'
import glob, os, re, sys
logs = sys.argv[1]
faults = []
for path in sorted(glob.glob(os.path.join(logs, "stderr_rank*.txt"))):
    rank = int(re.search(r'rank(\d+)', path).group(1))
    for line in open(path, errors="replace"):
        m = re.search(r'Memory access fault by GPU node-(\d+).*?on address '
                      r'(0x[0-9a-f]+)\. Reason: (.*)', line)
        if m:
            faults.append((rank, int(m.group(1)), int(m.group(2), 16),
                           m.group(3).strip()))
if not faults:
    print("  NO memory access faults in any rank stderr")
for rank, node, addr, reason in faults:
    print(f"\n  rank {rank}: GPU node-{node} (= device {node-2}) "
          f"faulted at 0x{addr:x} -- {reason}")
    # Which heap, and at what offset?
    table = os.path.join(logs, f"stdout_rank{rank}.txt")
    hit = False
    if os.path.exists(table):
        for line in open(table, errors="replace"):
            m = re.search(r'heap\[(\d+)\] = (0x[0-9a-f]+) \.\. (0x[0-9a-f]+)',
                          line)
            if m:
                i, lo, hi = int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16)
                if lo <= addr < hi:
                    off = addr - lo
                    print(f"      INSIDE heap[{i}] at byte offset {off} "
                          f"({off/2**20:.3f} MiB); as int index {off//4}, "
                          f"as bf16 index {off//2}")
                    hit = True
    mp = os.path.join(logs, f"maps_rank{rank}.txt")
    if os.path.exists(mp):
        for line in open(mp, errors="replace"):
            span = line.split()[0]
            lo, hi = (int(x, 16) for x in span.split('-'))
            if lo <= addr < hi:
                print(f"      maps region: {line.rstrip()}")
                print(f"      offset into region: {addr-lo} "
                      f"({(addr-lo)/2**20:.3f} MiB), region size "
                      f"{(hi-lo)/2**20:.1f} MiB")
                hit = True
        if not hit:
            print("      NOT inside any region present at stage-1 time -- "
                  "the mapping was created later or the address is bogus")
            # nearest regions
            best = []
            for line in open(mp, errors="replace"):
                span = line.split()[0]
                lo, hi = (int(x, 16) for x in span.split('-'))
                best.append((min(abs(addr-lo), abs(addr-hi)), line.rstrip()))
            for d, line in sorted(best)[:3]:
                print(f"      nearest (+-{d} B): {line}")
PYEOF

echo
echo "===== rank stderr tails for faulting ranks ====="
for r in 0 1 2 3 4 5 6 7; do
  f="$LOGS/stderr_rank$r.txt"
  if grep -q 'Memory access fault' "$f" 2>/dev/null; then
    echo "--- rank $r ---"; tail -30 "$f"
  fi
done
echo "===== DONE ====="
