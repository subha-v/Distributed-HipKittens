#!/usr/bin/env bash
# exp_36: final gate audit across every green point, then prove the node is idle
# and the lease is released.
set -u
python3 - "$HOME/e36/sensitivity_grid.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ok = [p for p in d["points"] if p["status"] == "ok"]
camp = [p for p in ok if p["measurement"] == "campaign"]
print(f"green points={len(ok)}  campaigns={len(camp)}  screens={len(ok)-len(camp)}")
bad = []
for p in ok:
    g = p["gates"]
    checks = {
        "mok_gate_all_pass": all(v["passed"] for v in g["mok_gate"].values()) and len(g["mok_gate"]) == 3,
        "control_fails": g["control_fails"] is True,
        "soak_600": (g["soak"] or {}).get("completed") == 600 and (g["soak"] or {}).get("total") == 600,
        "soak_pass": (g["soak"] or {}).get("passed") is True,
        "pperr0": g["pperr_max"] == 0,
        "poison_survivors0": g["poison_survivors_max"] == 0,
        "selftest": bool(g["poison_selftest"]) and all(g["poison_selftest"].values()),
        "eager_gate": (g["eager"] or {}).get("gate_ok") is True,
    }
    fails = [k for k, v in checks.items() if not v]
    if fails:
        bad.append((p["raw"]["tag"], p["config"], fails))
print("gate audit:", "ALL GREEN" if not bad else f"{len(bad)} POINTS WITH GAPS")
for b in bad:
    print("  ", b)
pp = sorted({(p["gates"]["pperr_max"]) for p in ok})
print("distinct pperr over all green points:", pp)
st = sorted({tuple(sorted(p["gates"]["poison_selftest"].items())) for p in ok})
print("poison selftest variants:", st[:3])
PY

echo
echo "=== node state ==="
echo "utc: $(date -u +%FT%TZ)  host: $(hostname)"
git -C "$HOME/Distributed-HipKittens" log --oneline -1
echo "dirty: [$(git -C "$HOME/Distributed-HipKittens" status --porcelain | head -5 | tr '\n' ';')]"
echo "--- rocm-smi --showpids ---"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "  "$1" "$2}'
echo "--- our driver/campaign processes ---"
pgrep -af 'screenT.sh|rc_T.sh|k0_mok' | cut -c1-120 || echo "  none"
echo "--- harness GPU lease dir ---"
if [ -d /tmp/k0_mok_synthetic_gpu_lock ]; then
  echo "  STILL PRESENT (stale) -- removing since the node is idle"
  rmdir /tmp/k0_mok_synthetic_gpu_lock 2>/dev/null && echo "  removed" || echo "  COULD NOT REMOVE"
else
  echo "  absent: lease released"
fi
echo "--- GPU utilisation ---"
/opt/rocm/bin/rocm-smi --showuse --csv 2>/dev/null | awk -F, '$1 ~ /^card[0-7]$/ {printf "  %s use=%s\n", $1, $2}'
echo "--- files written by exp_36 (node) ---"
du -sh "$HOME/e36" 2>/dev/null
ls -d "$HOME"/k0-mok-e36* 2>/dev/null | wc -l | xargs echo "  campaign output roots:"
exit 0
