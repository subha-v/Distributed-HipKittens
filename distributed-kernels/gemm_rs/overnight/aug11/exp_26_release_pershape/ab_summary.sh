#!/usr/bin/env bash
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_26_release_pershape
for f in ab_fwd ab_rev; do
  [ -f "$D/logs/$f.log" ] || continue
  echo "############################## $f ##############################"
  sed -n '/PER-SHAPE SUMMARY/,$p' "$D/logs/$f.log"
done
echo
echo "############## pooled across both passes ##############"
python3 - "$D/logs/ab_pershape_ab_fwd.json" "$D/logs/ab_pershape_ab_rev.json" <<'PY'
import json, statistics, sys
runs = [json.load(open(p)) for p in sys.argv[1:]]
arms = ["ps0", "ps1", "rg2c", "ps0b"]
keys = list(runs[0]["per_shape"])
print(f"{'#':>2} {'shape':<18}{'t/CTA':>6}  " +
      "".join(f"{a:>10}" for a in arms) + "   pooled best, both allocation orders")
for i, k in enumerate(keys):
    pooled = {}
    for a in arms:
        s = []
        for r in runs:
            s += r["per_shape"][k]["arms"][a]["samples"]
        pooled[a] = s
    ppc = runs[0]["per_shape"][k]["arms"]["_geometry"]["tiles_per_cta"]
    print(f"{i+1:>2} {k:<18}{ppc:>6}  " +
          "".join(f"{min(pooled[a]):>10.2f}" for a in arms))
print()
print(f"{'#':>2} {'shape':<18}  {'ps1-ps0':>9}{'rg2c-ps0':>10}{'null':>9}"
      f"   {'ps1 range':>18} {'ps0 range':>18} {'rg2c range':>18}")
for i, k in enumerate(keys):
    pooled = {a: sum((r["per_shape"][k]["arms"][a]["samples"] for r in runs), [])
              for a in arms}
    b = {a: min(pooled[a]) for a in arms}
    print(f"{i+1:>2} {k:<18}  "
          f"{(b['ps1']/b['ps0']-1)*100:>+8.2f}%{(b['rg2c']/b['ps0']-1)*100:>+9.2f}%"
          f"{(b['ps0b']/b['ps0']-1)*100:>+8.2f}%   "
          f"{min(pooled['ps1']):>8.1f}-{max(pooled['ps1']):<9.1f}"
          f"{min(pooled['ps0']):>8.1f}-{max(pooled['ps0']):<9.1f}"
          f"{min(pooled['rg2c']):>8.1f}-{max(pooled['rg2c']):<9.1f}")
PY
