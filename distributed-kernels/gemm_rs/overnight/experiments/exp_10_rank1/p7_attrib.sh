#!/usr/bin/env bash
# exp_10 probe 7 -- NO GPU. Attribute each fault address to a region using the
# CORRECT rank's map, and test the int32-truncation hypothesis numerically:
# is the faulting address hack + 2*int32(base_addr[i]) rather than
# hack + 2*base_addr[i]?
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
ARM=$ON/compbench/rank1
LOGS=$ARM/logs_bisect_s0

python3 - "$LOGS" <<'PYEOF'
import glob, os, re, sys
logs = sys.argv[1]

def rank_of(path):
    # NB: the arm directory is itself called "rank1", so anchor on the filename.
    return int(re.search(r'rank(\d+)\.txt$', os.path.basename(path)).group(1))

tables = {}   # rank -> dict(hack=..., heaps=[(base_addr, ptr)], a=..., b=...)
for path in sorted(glob.glob(os.path.join(logs, "stdout_rank*.txt"))):
    r = rank_of(path)
    info = {"heaps": {}}
    for line in open(path, errors="replace"):
        m = re.search(r'hack=(0x[0-9a-f]+)', line)
        if m:
            info["hack"] = int(m.group(1), 16)
        m = re.search(r'heap\[(\d+)\] = (0x[0-9a-f]+) \.\. (0x[0-9a-f]+).*'
                      r'base_addr=(-?\d+)', line)
        if m:
            info["heaps"][int(m.group(1))] = (
                int(m.group(2), 16), int(m.group(3), 16), int(m.group(4)))
        m = re.search(r'input a\.data_ptr\(\)=(0x[0-9a-f]+) b=(0x[0-9a-f]+)',
                      line)
        if m:
            info["a"] = int(m.group(1), 16)
            info["b"] = int(m.group(2), 16)
    tables[r] = info

def s32(v):
    v &= 0xFFFFFFFF
    return v - 2**32 if v >= 2**31 else v

print("===== per-rank heap tables =====")
for r in sorted(tables):
    t = tables[r]
    if "hack" not in t:
        print(f"rank {r}: no table (did not reach stage 1 print)")
        continue
    print(f"rank {r}: hack=0x{t['hack']:x} a=0x{t.get('a',0):x}")
    for i in sorted(t["heaps"]):
        lo, hi, ba = t["heaps"][i]
        print(f"   heap[{i}] 0x{lo:x} base_addr={ba:>12} "
              f"int32={s32(ba):>12} {'OVERFLOWS' if s32(ba)!=ba else ''}")

print()
print("===== faults, attributed with the CORRECT rank's map =====")
for path in sorted(glob.glob(os.path.join(logs, "stderr_rank*.txt"))):
    r = rank_of(path)
    for line in open(path, errors="replace"):
        m = re.search(r'Memory access fault by GPU node-(\d+).*?on address '
                      r'(0x[0-9a-f]+)\. Reason: (.*)', line)
        if not m:
            continue
        node, addr, reason = int(m.group(1)), int(m.group(2), 16), m.group(3)
        print(f"\nrank {r} (GPU node-{node} = device {node-2}) "
              f"faulted at 0x{addr:x}: {reason.strip()}")
        t = tables.get(r, {})
        # region from this rank's own map
        mp = os.path.join(logs, f"maps_rank{r}.txt")
        if os.path.exists(mp):
            found = False
            for ml in open(mp, errors="replace"):
                lo, hi = (int(x, 16) for x in ml.split()[0].split('-'))
                if lo <= addr < hi:
                    perms = ml.split()[1]
                    print(f"   region: {ml.rstrip()}")
                    print(f"   -> perms={perms}  size={(hi-lo)/2**30:.2f} GiB"
                          f"  offset={(addr-lo)/2**20:.3f} MiB")
                    if perms.startswith('---'):
                        print("   -> NO-PERMISSION reserved VA. A GPU write "
                              "here reports 'read-only page'.")
                    found = True
            if not found:
                print("   region: address is in an unmapped gap")
        if "hack" not in t:
            continue
        hack = t["hack"]
        print(f"   addr - hack = {addr - hack} bytes "
              f"(= {(addr-hack)//2} bf16 elements)")
        # Which heap does the CORRECT arithmetic reach?
        for i in sorted(t["heaps"]):
            lo, hi, ba = t["heaps"][i]
            correct = hack + 2 * ba
            trunc = hack + 2 * s32(ba)
            for label, cand in (("exact", correct), ("int32-trunc", trunc)):
                delta = addr - cand
                if 0 <= delta < 2**31:
                    print(f"   heap[{i}] {label:12s} base=0x{cand:x} -> "
                           f"fault is +{delta} B (+{delta/2**20:.3f} MiB) "
                           f"{'<-- INSIDE 1 GiB heap' if delta < 2**30 else ''}")
PYEOF

echo
echo "===== compiled ISA: how are the heap_base constants materialized? ====="
CACHE=$ARM/.triton
echo "cache dirs: $(find "$CACHE" -maxdepth 1 -type d | wc -l)"
K=$(find "$CACHE" -name '*.json' -path '*kernel1*' 2>/dev/null | head -1)
echo "--- look for the mainloop kernel artifacts ---"
find "$CACHE" -name '*_kernel1*' 2>/dev/null | head -20
echo
echo "--- all amdgcn/llir files, largest first ---"
find "$CACHE" -name '*.amdgcn' -o -name '*.llir' -o -name '*.ttgir' 2>/dev/null \
  | xargs -r ls -S 2>/dev/null | head -10

BIG=$(find "$CACHE" -name '*.amdgcn' 2>/dev/null | xargs -r ls -S 2>/dev/null | head -1)
if [ -n "$BIG" ]; then
  echo
  echo "--- $BIG : global_store instructions and 64-bit address setup ---"
  grep -cE 'global_store' "$BIG"
  echo "--- s_mov_b32 with big constants (the heap_base literals) ---"
  grep -oE 's_mov_b32 s[0-9]+, (0x[0-9a-f]+|-?[0-9]+)' "$BIG" | sort | uniq -c | sort -rn | head -25
fi
echo "===== DONE p7 ====="
