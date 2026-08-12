#!/usr/bin/env bash
# exp_27: roll up all three campaigns from their summary.json files.
set -uo pipefail
python3 - <<'PY'
import glob, json, os
ARM = {"e27camp_cand": 1, "e27camp2_ctl": 0, "e27camp3_cand": 1}
print(f"{'campaign':16s} {'arm':>3s} {'production':>11s} {'pf6gm':>9s} {'mps_mega':>9s} "
      f"{'mps/prod':>8s} {'mps/pf6gm':>9s} {'pf6gm/prod':>10s}")
out = {}
for tag, arm in ARM.items():
    for sj in sorted(glob.glob(os.path.expanduser(f"~/k0-mok-{tag}/*/summary.json"))):
        d = json.load(open(sj)); ap = d.get("arm_p50_us", {})
        g = lambda a: ap.get(a, {}).get("median")
        p, f, m = g("production"), g("pf6gm_mega"), g("mps_mega")
        if not (p and f and m): continue
        out.setdefault(arm, []).append(m)
        print(f"{tag:16s} {arm:3d} {p:11.1f} {f:9.1f} {m:9.1f} {m/p:8.4f} {m/f:9.4f} {f/p:10.4f}")
        print(f"{'':16s}     rotations mps = {[round(x,1) for x in ap['mps_mega']['values']]}")
print()
if 0 in out and 1 in out:
    c = sum(out[0])/len(out[0]); k = sum(out[1])/len(out[1])
    print(f"control  (arm 0) mean of {len(out[0])} campaign(s) = {c:8.1f} us")
    print(f"candidate(arm 1) mean of {len(out[1])} campaign(s) = {k:8.1f} us   spread = "
          f"{max(out[1])-min(out[1]):.1f} us")
    print(f"==> same-session paired delta = {k-c:+.1f} us  ({(k-c)/c*100:+.2f}%)")
    print(f"==> vs the PUBLISHED ratchet 6568.0 us = {k-6568.0:+.1f} us")
PY
