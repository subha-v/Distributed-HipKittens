#!/usr/bin/env bash
# exp_24 close-out: node left clean, all campaign artifacts present.
set -uo pipefail
echo "=== node idle"
pgrep -af 'torchrun|mpirun|screen.sh|run_campaign' | head
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
echo "=== stale GPU lease?"
ls -ld /tmp/k0_mok_synthetic_gpu_lock 2>&1 | head -1
echo "=== stale JIT lock? (the 25-min-timeout trap from LESSONS)"
docker exec subha_k1 find /home/subvadla/.cache/k0-mok-synthetic-prefill/mori/jit/ -name '*.lock' 2>/dev/null | head
echo "=== campaign csvs"
for t in e24a e24b e24c e24d e24e; do
  f=$HOME/overnight-scratch/screen_$t.csv
  [ -f "$f" ] && printf "%s: %d rows\n" "$t" "$(( $(wc -l < "$f") - 1 ))"
done
echo "=== every mps_us measured tonight, by config"
python3 - <<'PY'
import csv, glob, os, statistics as st
byc = {}
for p in sorted(glob.glob(os.path.expanduser("~/overnight-scratch/screen_e24*.csv"))):
    rows = list(csv.reader(open(p)))
    h = rows[0]; i = {k: h.index(k) for k in h}
    for r in rows[1:]:
        if r[i["status"]] != "OK": continue
        key = (r[i["cfg"]], r[i["iters"]])
        try: byc.setdefault(key, []).append(float(r[i["mps_us"]]))
        except ValueError: pass
for (cfg, it), v in sorted(byc.items(), key=lambda kv: st.mean(kv[1])):
    s = f"{st.stdev(v):6.1f}" if len(v) > 1 else "     -"
    print(f"{it:10s} n={len(v):2d} mean={st.mean(v):8.1f} sd={s}  {cfg}")
PY
echo "=== node checkout HEAD (should be the exp_24 result commit after a resync)"
git -C "$HOME/Distributed-HipKittens" fetch -q --all
git -C "$HOME/Distributed-HipKittens" reset -q --hard origin/codex/distributed-hipkittens-scaffold
git -C "$HOME/Distributed-HipKittens" log --oneline -1
grep -oE 'K0P6_MPS_SRC_REV [0-9]+' \
  "$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip" | tail -1
exit 0
