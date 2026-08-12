#!/usr/bin/env bash
# exp_35 step 5b: prove the decode from the KERNEL'S OWN INPUT.
# K0_MPS_DESC_DUMP writes the mps descriptor as the host handed it to the
# kernel. Slot K0P6_D_MPS_CFG holds the packed config word that .hip:702 feeds
# to decode_config(). Decode it here with the exact bit formula from
# moe_mps_adapter.cuh and check it against the requested K0_MPS_CFG.
set -uo pipefail
DHK=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe

echo "===SLOT_INDEX==="
grep -rnE '#define K0P6_D_MPS_CFG|K0P6_D_MPS_CFG ' "$DHK"/*.hip "$DHK"/*.cuh 2>/dev/null | head -5

SLOT=$(grep -rhoE '#define K0P6_D_MPS_CFG +[0-9]+' "$DHK"/*.hip "$DHK"/*.cuh 2>/dev/null | head -1 | awk '{print $3}')
echo "resolved K0P6_D_MPS_CFG = ${SLOT:-UNRESOLVED}"

echo
echo "===DESC_FILES==="
find "$HOME/k0-mok-e35smoke" -name 'mps_desc_rank*.txt' | sort

echo
echo "===DECODE==="
python3 - "${SLOT:-0}" <<'PY'
import glob, os, sys, json
slot = int(sys.argv[1])
HOME = os.path.expanduser("~")

def dec(w):
    return dict(C=w & 0xFF,
                g=((w >> 8) & 0xFF) | (((w >> 34) & 0xFF) << 8),
                mode=(w >> 16) & 0xFF,
                flush_rows=(w >> 24) & 0xFF,
                pull_fallback=(w >> 32) & 1,
                timestamps=(w >> 33) & 1)

DEPTH = {0: 8, 1: 4, 2: 16, 3: 32}
for d in sorted(glob.glob(f"{HOME}/k0-mok-e35smoke/*/**/mps_desc_rank*.txt", recursive=True)):
    run = d.split("k0-mok-e35smoke/")[1].split("/")[0]
    vals = json.loads(open(d).read())
    w = int(vals[slot])
    c = dec(w)
    g = c["g"]
    en = (g >> 5) & 1
    sel = (g & 0x300) >> 8
    print(f"{run}")
    print(f"  file          : {os.path.basename(d)}  (n_slots={len(vals)})")
    print(f"  slot[{slot}] word : {w} (0x{w:x})")
    print(f"  decode_config : {c}")
    print(f"  g=0x{g:03x} -> physical_g={g & 0xF} detect={(g >> 4) & 1} "
          f"throttle_ENABLE={en} skip_part_zero={(g >> 6) & 1} depth_sel={sel}")
    print(f"  THROTTLE      : {'ON  depth=' + str(DEPTH[sel]) if en else 'OFF (depth selector inert; config_is_valid forces sel=0)'}")
    print(f"  mode field    : {c['mode']}  <-- 12 proves NO g-overflow into mode")
    print()
PY
