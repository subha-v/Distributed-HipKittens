#!/usr/bin/env bash
# exp_34 step 12: (a) finish the attribution with two VALID region arms, and
# (b) test the candidate fix -- encode "readiness already settled" as
# `row_ready == nullptr` on the EXISTING parameter instead of a new template
# parameter, so mode 14 shares mode 12's single M8 instantiation and no fifth
# inlined combine body is created.
#   V_a  = base + m8_batch template + M8 phase changes
#   V_b  = base + the M7.5 block changes
#   V_rt = full candidate, runtime nullptr encoding, hooks reduced to the one
#          task_drain token, M8 coarse branch folded into the direct-accum branch
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out

python3 - "$SC" <<'PY'
import sys, pathlib, difflib
sc = pathlib.Path(sys.argv[1])
base = (sc/"hunk/base.hip").read_text().splitlines(keepends=True)
n8   = (sc/"hunk/n8.hip").read_text().splitlines(keepends=True)
sm = difflib.SequenceMatcher(None, base, n8, autojunk=False)
ops = [o for o in sm.get_opcodes() if o[0] != 'equal']
def build(sel):
    out, cur = [], 0
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        out.extend(base[cur:i1]); out.extend(n8[j1:j2] if k in sel else base[i1:i2]); cur = i2
    out.extend(base[cur:]); return "".join(out)
arms = {"V_a": {0,1,6,7,8,9,16,17,18,19,20},
        "V_b": {0,1,11,12,13,14,15}}
for name, sel in arms.items():
    d = sc/"tu"/name; d.mkdir(parents=True, exist_ok=True)
    (d/"k0pf6gm_device_tile_mps.hip").write_text(build(sel))
    print(name, "written")

# ---------------- V_rt ----------------
src = (sc/"tu/N8/k0pf6gm_device_tile_mps.hip").read_text()
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:70], s.count(a))
    return s.replace(a, b)

# 1. drop the Ready template parameter; gate on row_ready == nullptr instead
src = rep(src,
  "         bool Detect = false, bool Zero = false, bool Ready = false>",
  "         bool Detect = false, bool Zero = false>")
src = rep(src,
  """        if constexpr (!Ready) {
          batch_ready = hk_moe::poll_epoch_system(
              row_ready + (size_t)p * T_loc_max + row, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, 33554432});
        }""",
  """        if (row_ready != nullptr) {
          batch_ready = hk_moe::poll_epoch_system(
              row_ready + (size_t)p * T_loc_max + row, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, 33554432});
        }""")
src = rep(src,
  """  if constexpr (!Ready) {
    if (__ballot(batch_ready) != ~0ull) return;
  } else {
    (void)batch_ready;
  }""",
  """  if (row_ready != nullptr && __ballot(batch_ready) != ~0ull) return;""")
# 2. delete the separate coarse M8 branch; route mode 14 through direct-accum
i = src.index("      } else if (m8_coarse) {")
j = src.index("      } else if (hk_moe::mps::mode_is_direct_accum(m8cfg)) {")
src = src[:i] + src[j:]
src = rep(src,
  "              row_ready, T_loc_max, pperr, lane, cur, out, base_slot, base_slot);\n        }\n      } else {",
  "              m8_coarse ? nullptr : row_ready,\n"
  "              T_loc_max, pperr, lane, cur, out, base_slot, base_slot);\n        }\n      } else {")
# 3. hooks: keep ONLY the task_drain token
src = rep(src, "  if (k0p6_m == 14ull) return;\n", "")
src = rep(src, "  if (m == 14ull) return;\n", "")
d = sc/"tu/V_rt"; d.mkdir(parents=True, exist_ok=True)
(d/"k0pf6gm_device_tile_mps.hip").write_text(src)
print("V_rt written", len(src))
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
for v in V_a V_b V_rt; do
  hipcc "${COMMON[@]}" --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  rc=$?
  hipcc "${COMMON[@]}" --cuda-device-only -S $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
  echo "$v exit=$rc errors=$(grep -cE "error:" $O/$v.log)"
  [ $rc -ne 0 ] && grep -m4 -E "error:" $O/$v.log
done
echo built'

echo
printf '%-6s %-8s %-9s %-9s %-9s %-8s %-8s %-14s\n' arm scratch sgpr_sp vgpr_sp scr_load mfma pk_add exp26_sig
for v in A W2 Z_all V_a V_b V_rt; do
  f=$O/$v.log; s=$O/$v.s
  [ -s "$f" ] || continue
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  sig="-"
  [ -s "$s" ] && sig=$(awk '
      /^[[:space:]]+flat_atomic_pk_add_bf16/ { ++na; aline[na]=NR }
      /^[[:space:]]+scratch_(load|store)/    { ++ns; sline[ns]=NR }
      END { h=0; for (i=1;i<=na;i++) for (j=1;j<=ns;j++) if (sline[j]<aline[i] && aline[i]-sline[j]<=40) { h++; break }
            printf "%d/%d", h, na }' "$s")
  printf '%-6s %-8s %-9s %-9s %-9s %-8s %-8s %-14s\n' "$v" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" \
    "$(c scratch_load)" "$(c v_mfma)" "$(c flat_atomic_pk_add_bf16)" "$sig"
done
echo "===DONE==="
