#!/usr/bin/env bash
# exp_34 step 11: hunk-level bisect of the scratch migration. Split the kernel
# diff (base ca5b683f -> N8 shape) into hunks, label each by the region it lands
# in, and build one arm per region on top of the pristine base.
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
mkdir -p "$SC/hunk"

cp "$SC/base/k0pf6gm_device_tile_mps.hip.orig" "$SC/hunk/base.hip"
cp "$SC/tu/N8/k0pf6gm_device_tile_mps.hip"     "$SC/hunk/n8.hip"

python3 - "$SC" <<'PY'
import sys, pathlib, difflib, re
sc = pathlib.Path(sys.argv[1])
base = (sc/"hunk/base.hip").read_text().splitlines(keepends=True)
n8   = (sc/"hunk/n8.hip").read_text().splitlines(keepends=True)
sm = difflib.SequenceMatcher(None, base, n8, autojunk=False)
ops = [o for o in sm.get_opcodes() if o[0] != 'equal']
print(f"{len(ops)} change groups")
labels = []
for i, (tag, i1, i2, j1, j2) in enumerate(ops):
    ctx = "".join(n8[j1:j2])[:120].replace("\n", " ")
    print(f"  g{i}: {tag} base[{i1}:{i2}] -> n8[{j1}:{j2}]  {ctx}")
    labels.append((tag, i1, i2, j1, j2))

def build(sel):
    out, cur = [], 0
    for k, (tag, i1, i2, j1, j2) in enumerate(labels):
        out.extend(base[cur:i1])
        out.extend(n8[j1:j2] if k in sel else base[i1:i2])
        cur = i2
    out.extend(base[cur:])
    return "".join(out)

# region assignment by BASE line number (0-indexed):
#   guard  : the entry guard  (~line 700-725)
#   hooks  : per-task hooks   (~line 300-420)
#   m8b    : k0p6_mps_m8_batch template + if constexpr  (~line 480-580)
#   m75    : the M7.5 block   (>1600)
#   m8     : the M8 phase     (>1740)
#   hdr    : header/SRC_REV/comments (<300)
regions = {}
for k, (tag, i1, i2, j1, j2) in enumerate(labels):
    if i1 < 300:      r = "hdr"
    elif i1 < 460:    r = "hooks"
    elif i1 < 640:    r = "m8b"
    elif i1 < 760:    r = "guard"
    elif i1 < 1660:   r = "m75pre"
    elif i1 < 1740:   r = "m75"
    else:             r = "m8"
    regions.setdefault(r, []).append(k)
print("regions:", {k: v for k, v in regions.items()})

# Arms: base + one region at a time (always include hdr, which is comments only)
hdr = set(regions.get("hdr", []))
arms = {}
for r, ks in regions.items():
    if r == "hdr": continue
    arms["Z_"+r] = hdr | set(ks)
arms["Z_all"] = set(range(len(labels)))
# and the combination we suspect is required for a legal build
for name, sel in arms.items():
    d = sc/"tu"/name
    d.mkdir(parents=True, exist_ok=True)
    (d/"k0pf6gm_device_tile_mps.hip").write_text(build(sel))
    print(name, "->", sorted(sel))
PY

docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla; SC=$H/e34; O=$SC/out
DHK=$SC/DHK; FM=$DHK/distributed-kernels/fused_moe
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for d in $SC/tu/Z_*; do
  v=$(basename $d)
  hipcc "${COMMON[@]}" --genco $d/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  rc=$?
  hipcc "${COMMON[@]}" --cuda-device-only -S $d/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
  echo "$v exit=$rc errors=$(grep -cE "error:" $O/$v.log)"
done
echo built'

echo
printf '%-12s %-8s %-11s %-11s %-9s %-24s\n' arm scratch sgpr_sp vgpr_sp scr_load exp26_signature
for v in A W2 $(cd $SC/tu && ls -d Z_* 2>/dev/null); do
  f=$O/$v.log; s=$O/$v.s
  [ -s "$f" ] || continue
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  sig="-"
  if [ -s "$s" ]; then
    sig=$(awk '
      /^[[:space:]]+v_mfma/                  { ++nm; mline[nm]=NR }
      /^[[:space:]]+flat_atomic_pk_add_bf16/ { ++na; aline[na]=NR }
      /^[[:space:]]+scratch_(load|store)/    { ++ns; sline[ns]=NR }
      END { h=0
        for (i=1;i<=na;i++) for (j=1;j<=ns;j++) if (sline[j]<aline[i] && aline[i]-sline[j]<=40) { h++; break }
        printf "%d/%d", h, na }' "$s")
  fi
  printf '%-12s %-8s %-11s %-11s %-9s %-24s\n' "$v" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" "$(c scratch_load)" "$sig"
done
echo "===DONE==="
