#!/usr/bin/env bash
# exp_35 step 4: (a) the device-side config_is_valid gate, (b) where screen.sh
# puts each run's output root, (c) the expected packed cfg words for the two
# rungs so the DESC_DUMP can be checked by arithmetic rather than by eye.
set -uo pipefail
MPS=$HOME/Distributed-HipKittens/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

echo "===DEVICE_VALIDITY_GATE (.hip 694-730)==="
sed -n '694,730p' "$MPS"

echo
echo "===SCREEN_OUT_ROOTS==="
grep -nE '^OUT=|^LOG=|OUT="|LOG="|HEADSHA|SRCREV|hsaco_stamp\(\)|node_idle\(\)' "$HOME/tools/screen.sh" | head -30

echo
echo "===EXPECTED_PACKED_WORDS==="
python3 - <<'PY'
def enc(C, g, mode, flush, pull=0, ts=0):
    return (C
            | ((g & 0xFF) << 8)
            | (mode << 16)
            | (flush << 24)
            | (pull << 32)
            | (ts << 33)
            | (((g >> 8) & 0xFF) << 34))

def dec(w):
    return dict(C=w & 0xFF,
                g=((w >> 8) & 0xFF) | (((w >> 34) & 0xFF) << 8),
                mode=(w >> 16) & 0xFF,
                flush=(w >> 24) & 0xFF,
                pull=(w >> 32) & 1,
                ts=(w >> 33) & 1)

for label, g in (("rung_c_ratchet_depth4", 353), ("rung_b_unthrottled", 65),
                 ("depth8_A_only(ref)", 97), ("ILLEGAL_depth4_no_enable", 321)):
    for ts in (0, 1):
        w = enc(16, g, 12, 16, 0, ts)
        d = dec(w)
        ok = (d["g"] == g and d["mode"] == 12 and d["C"] == 16 and d["flush"] == 16)
        print(f"{label:26s} g={g:4d} (0x{g:03x}) ts={ts}  word={w} (0x{w:x})  "
              f"decodes->{d}  roundtrip_ok={ok}")
    print(f"{'':26s}   bits: physical_g={g & 0xF} detect={(g >> 4) & 1} "
          f"throttle_en={(g >> 5) & 1} skip_part_zero={(g >> 6) & 1} "
          f"depth_sel={(g & 0x300) >> 8} -> depth="
          f"{ {0: 8, 1: 4, 2: 16, 3: 32}[(g & 0x300) >> 8] }")
    print()
PY
# end
