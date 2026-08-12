#!/usr/bin/env bash
# Final rematch report: the post-fix aggregate, the archived pre-fix aggregate
# for the before/after, and an explicit dump of the bias flags + correctness
# from every per-shape JSON (a speed number from an incorrect arm is worthless).
set -u
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
E=$ON/experiments/exp_10_rank1

echo "##################### POST-FIX (this run) #####################"
python3 "$E/vs_report.py" "$E/vs_logs"

echo
echo "##################### bias flags per shape, all 8 ranks #####################"
python3 - "$E/vs_logs" <<'PY'
import glob, json, os, sys, collections
root = sys.argv[1]
rows = collections.defaultdict(lambda: {"forced": set(), "present": set(), "hb": set(), "n": 0})
for p in sorted(glob.glob(os.path.join(root, "vs_s*.rank*.json"))):
    idx = int(os.path.basename(p).split(".")[0].replace("vs_s", ""))
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not d:
        continue
    r = rows[idx]
    r["forced"].add(bool(d.get("bias_forced")))
    r["present"].add(bool(d.get("bias_present")))
    r["hb"].add(int(d["shape"][3]))
    r["n"] += 1
    r["shape"] = "x".join(str(v) for v in d["shape"][:3])
print(f"{'#':>2} {'shape':>21} {'ranks':>6} {'declared_bias':>14} "
      f"{'bias_forced':>12} {'bias_present':>13}")
for idx in sorted(rows):
    r = rows[idx]
    fmt = lambda s: ",".join(str(x) for x in sorted(s, key=str))
    print(f"{idx+1:>2} {r['shape']:>21} {r['n']:>6} {fmt(r['hb']):>14} "
          f"{fmt(r['forced']):>12} {fmt(r['present']):>13}")
print()
print("every row must read bias_forced=True and bias_present=True: that is what")
print("the official evaluator actually feeds, and it is the condition under")
print("which the tuned shape-table rows must match.")
PY

echo
echo "##################### PRE-FIX archive (for the before/after) #####################"
for d in "$E"/vs_logs_prefix_*; do
  [ -d "$d" ] || continue
  echo "--- $d ---"
  python3 - "$d" <<'PY'
import glob, json, os, statistics, sys, math
root = sys.argv[1]
sh = {}
for p in sorted(glob.glob(os.path.join(root, "vs_s*.rank*.json"))):
    idx = int(os.path.basename(p).split(".")[0].replace("vs_s", ""))
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not d:
        continue
    sh.setdefault(idx, {"ours": [], "rank1": [], "forced": set()})
    sh[idx]["ours"] += d["arms"].get("ours", [])
    sh[idx]["rank1"] += d["arms"].get("rank1", [])
    sh[idx]["forced"].add(bool(d.get("bias_forced")))
    sh[idx]["shape"] = "x".join(str(v) for v in d["shape"][:3])
def geo(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
gb = {"ours": [], "rank1": []}
gm = {"ours": [], "rank1": []}
print(f"{'#':>2} {'shape':>21} {'ours best':>10} {'ours mean':>10} "
      f"{'r1 best':>10} {'r1 mean':>10} {'forced':>7}")
for idx in sorted(sh):
    r = sh[idx]
    if not r["ours"] or not r["rank1"]:
        continue
    for a in ("ours", "rank1"):
        gb[a].append(min(r[a])); gm[a].append(statistics.mean(r[a]))
    print(f"{idx+1:>2} {r['shape']:>21} {min(r['ours']):10.2f} "
          f"{statistics.mean(r['ours']):10.2f} {min(r['rank1']):10.2f} "
          f"{statistics.mean(r['rank1']):10.2f} "
          f"{','.join(str(x) for x in sorted(r['forced'], key=str)):>7}")
if gb["ours"]:
    print(f"  geo best  ours={geo(gb['ours']):.2f} rank1={geo(gb['rank1']):.2f} "
          f"ratio={geo(gb['ours'])/geo(gb['rank1']):.3f}x  (over these shapes only)")
    print(f"  geo mean  ours={geo(gm['ours']):.2f} rank1={geo(gm['rank1']):.2f} "
          f"ratio={geo(gm['ours'])/geo(gm['rank1']):.3f}x")
PY
done
echo "===== DONE ====="
