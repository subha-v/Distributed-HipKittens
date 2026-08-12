#!/usr/bin/env bash
# exp_34 mechanism check, part 2. rev 26 shows the four compile-time throttle
# instantiations in ISA: vmcnt(4)=96, vmcnt(8)=96, vmcnt(16)=96, vmcnt(32)=96.
# Emit the same census at the pin. The scratch clone predates the pin, so fetch.
set -uo pipefail
SC=$HOME/e34
git -C "$SC/DHK" fetch -q --all
git -C "$SC/DHK" rev-parse --verify 291dfa08^{commit}
rm -rf "$SC/isa/291dfa08" "$SC/isa/291dfa08.s"
mkdir -p "$SC/isa/291dfa08"
git -C "$SC/DHK" archive 291dfa08 | tar -x -C "$SC/isa/291dfa08"
grep -m1 "define K0P6_MPS_SRC_REV" "$SC/isa/291dfa08/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"

docker exec subha_k1 bash -lc '
set -uo pipefail
SC=/home/subvadla/e34
K0=/home/subvadla/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
REV=291dfa08
DHK=$SC/isa/$REV
FM=$DHK/distributed-kernels/fused_moe
t0=$SECONDS
hipcc -S --cuda-device-only --offload-arch=gfx950 -std=c++20 -O3 \
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
  -I$DHK/include -I$FM -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels \
  -I$K0/solution/hip -I$MR -I$MR/include -I$MR/src \
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
  $FM/k0pf6gm_device_tile_mps.hip -o $SC/isa/$REV.s > $SC/isa/$REV.log 2>&1
echo "$REV exit=$? secs=$((SECONDS-t0)) lines=$(wc -l < $SC/isa/$REV.s 2>/dev/null || echo 0) errors=$(grep -cE "error:" $SC/isa/$REV.log)"
grep -E "error:" $SC/isa/$REV.log | head -5
'
echo
echo "===BOUNDED vmcnt census, both revisions==="
python3 - <<'PY'
import os, re, collections
SC = os.path.expanduser("~/e34/isa")
for rev in ("f113d73f", "291dfa08"):
    p = os.path.join(SC, rev + ".s")
    if not os.path.exists(p):
        print(rev, "MISSING"); continue
    txt = open(p, errors="ignore").read()
    waits = re.findall(r"s_waitcnt\s+vmcnt\((\d+)\)", txt)
    c = collections.Counter(int(w) for w in waits)
    bounded = {k: v for k, v in sorted(c.items()) if k != 0}
    print(f"{rev}: lines={txt.count(chr(10))} total vmcnt={len(waits)} "
          f"vmcnt(0)={c[0]} BOUNDED={bounded}")
PY
echo "===DONE==="
