#!/usr/bin/env bash
# exp_34 mechanism check for the throttle regression. The exp_24 injection bound
# compiles to BOUNDED `s_waitcnt vmcnt(N)` (N>0) waits in the phase-2 push
# epilogue -- "one of four compile-time instantiations, never a runtime vmcnt".
# Mode 12's source path is byte-identical between rev 26 and the pin, yet mode 12
# lost the throttle's 620-720 us at the pin. So compare the BOUNDED-vmcnt census
# of the two builds. CPU only, in the private scratch clone; the node checkout and
# the running campaign are untouched.
set -uo pipefail
SC=$HOME/e34
mkdir -p "$SC/isa"
for REV in f113d73f 291dfa08; do
  D=$SC/isa/$REV
  if [ ! -d "$D" ]; then
    mkdir -p "$D"
    git -C "$SC/DHK" archive "$REV" | tar -x -C "$D"
  fi
  echo "extracted $REV -> $D  ($(grep -m1 'define K0P6_MPS_SRC_REV' "$D/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"))"
done

docker exec subha_k1 bash -lc '
set -uo pipefail
SC=/home/subvadla/e34
K0=/home/subvadla/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
for REV in f113d73f 291dfa08; do
  DHK=$SC/isa/$REV
  FM=$DHK/distributed-kernels/fused_moe
  [ -s $SC/isa/$REV.s ] && { echo "### $REV .s cached"; continue; }
  echo "### emitting ISA for $REV"
  t0=$SECONDS
  hipcc -S --cuda-device-only --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -I$DHK/include -I$FM -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels \
    -I$K0/solution/hip -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $FM/k0pf6gm_device_tile_mps.hip -o $SC/isa/$REV.s > $SC/isa/$REV.log 2>&1
  echo "$REV exit=$? secs=$((SECONDS-t0)) lines=$(wc -l < $SC/isa/$REV.s 2>/dev/null || echo 0) errors=$(grep -cE "error:" $SC/isa/$REV.log)"
done
'
echo
echo "===BOUNDED vmcnt census (N>0 = the throttle; vmcnt(0) = ordinary drains)==="
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
    print(f"{rev}: total vmcnt waits={len(waits)}  vmcnt(0)={c[0]}  BOUNDED(N>0)={bounded}")
PY
echo "===DONE==="
