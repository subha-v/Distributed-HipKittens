#!/usr/bin/env bash
# exp_34 step 13: the M7.5 block is where additions re-trigger exp_26's scratch
# migration (V_b alone = 186/282); the M8 region tolerates them (V_a = 0/282).
#   V_c = publish FOLDED INTO the parity publication loop (two selects instead of
#         a second arm); poll still in M7.5.
#   V_d = V_c + the poll+converge moved to the HEAD OF M8, where V_a showed
#         additions are free.
#   V_e = V_d + the hooks reduced to the one task_drain token.
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out

python3 - "$SC" <<'PY'
import sys, pathlib
sc = pathlib.Path(sys.argv[1])
src = (sc/"tu/N8/k0pf6gm_device_tile_mps.hip").read_text()
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)

PUB_ARM = """      if (coarse75) {
        if (bid == 0) {
          unsigned int* self_slot =
              row_ready + (size_t)cur * T_loc_max + (size_t)T_ext;
          for (int R = 0; R < world; ++R) {
            if (R == cur) {
              hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);
            } else {
              hk_moe::publish_epoch<hk_moe::scope::system>(
                  hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
            }
          }
        }
      } else {
      // Each CTA leader owns r = bid (mod nct) over the receive-row EXTENT. row_remaining!=0
      // selects live rows; holes are 0 and skipped.
      for (int r = bid; r < T_ext; r += nct) {
        if (hk_moe::load_relaxed<hk_moe::scope::agent>(row_remaining + r) == 0u)
          continue;   // hole row: nothing certified it in M2, no combine reads it
"""
PUB_FOLDED = """      // exp_34 mode 14 reuses this loop rather than adding a second arm: it
      // publishes ONE cell (index T_ext, CTA 0 only) with the same stores, the
      // same scopes and the same peer table. Two selects, no new code region --
      // a separate arm here re-triggered exp_26's epilogue scratch migration.
      const int r_first = coarse75 ? (bid == 0 ? T_ext : T_ext + nct) : bid;
      const int r_end = coarse75 ? T_ext + 1 : T_ext;
      for (int r = r_first; r < r_end; r += nct) {
        if (!coarse75 &&
            hk_moe::load_relaxed<hk_moe::scope::agent>(row_remaining + r) == 0u)
          continue;   // hole row: nothing certified it in M2, no combine reads it
"""
CLEAN = """        hk_moe::publish_relaxed<hk_moe::scope::agent>(row_remaining + r, 0u);
      }
      }
    }
"""
CLEAN_FOLDED = """        if (!coarse75) {
          hk_moe::publish_relaxed<hk_moe::scope::agent>(row_remaining + r, 0u);
        }
      }
    }
"""
POLL_M75 = """    if (coarse75) {
      if (payload_ok && tid == 0) {
        for (int p = 0; p < world; ++p) {
          (void)hk_moe::poll_epoch_system(
              row_ready + (size_t)p * T_loc_max + (size_t)T_ext, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
        }
      }
      __syncthreads();
    }
"""
POLL_M8 = """    // exp_34 stage (4)+(5): the cross-rank rendezvous. It lives at the HEAD OF M8
    // rather than beside the publish for a measured reason: additions to the M7.5
    // region re-trigger exp_26's scratch migration into the remote-atomic
    // epilogue, and additions here do not. It must precede `payload_ok` so a
    // rendezvous timeout (bit 25) suppresses the combine in the same phase.
    if (m8_coarse) {
      if (tid == 0 &&
          !hk_moe::error_bit_set_agent(pperr, 16777216 | 2097152)) {
        const unsigned int* m7d =
            (const unsigned int*)k0p6_dread(desc, K0P6_D_ROW_READY);
        for (int p = 0; p < world; ++p) {
          (void)hk_moe::poll_epoch_system(
              m7d + (size_t)p * (size_t)T_loc_max + (size_t)T_ext, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
        }
      }
      // Unconditional in this block: `m8_coarse` is a grid-uniform decode of one
      // descriptor word, so every thread of the CTA reaches this barrier. It is
      // what orders "some thread observed all 8 flags" before ANY thread of this
      // CTA reads a slot.
      __syncthreads();
    }
"""
v_c = rep(rep(src, PUB_ARM, PUB_FOLDED), CLEAN, CLEAN_FOLDED)
v_d = rep(v_c, POLL_M75, "")
v_d = rep(v_d,
  "    const bool payload_ok = !hk_moe::error_bit_set_agent(\n"
  "        pperr, 16777216 | 2097152 |\n"
  "        (m8_dynamic ? (K0P6_MPS_ERR_SERVICE | K0P6_MPS_ERR_CONFIG) : 0) |\n"
  "        (m8_coarse ? K0P6_MPS_ERR_M7DONE : 0));",
  POLL_M8 +
  "    const bool payload_ok = !hk_moe::error_bit_set_agent(\n"
  "        pperr, 16777216 | 2097152 |\n"
  "        (m8_dynamic ? (K0P6_MPS_ERR_SERVICE | K0P6_MPS_ERR_CONFIG) : 0) |\n"
  "        (m8_coarse ? K0P6_MPS_ERR_M7DONE : 0));")
v_e = rep(rep(v_d, "  if (k0p6_m == 14ull) return;\n", ""), "  if (m == 14ull) return;\n", "")
for name, text in (("V_c", v_c), ("V_d", v_d), ("V_e", v_e)):
    d = sc/"tu"/name; d.mkdir(parents=True, exist_ok=True)
    (d/"k0pf6gm_device_tile_mps.hip").write_text(text)
    print(name, "written", len(text))
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
for v in V_c V_d V_e; do
  hipcc "${COMMON[@]}" --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  rc=$?
  hipcc "${COMMON[@]}" --cuda-device-only -S $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
  echo "$v exit=$rc errors=$(grep -cE "error:" $O/$v.log)"
  [ $rc -ne 0 ] && grep -m4 -E "error:" $O/$v.log
done
echo built'

echo
printf '%-6s %-8s %-9s %-9s %-9s %-8s %-8s %-10s %-14s\n' \
  arm scratch sgpr_sp vgpr_sp scr_load mfma pk_add in_mfma exp26_sig
for v in A W2 Z_all V_c V_d V_e; do
  f=$O/$v.log; s=$O/$v.s
  [ -s "$f" ] || continue
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  inm="-"; sig="-"
  if [ -s "$s" ]; then
    read inm sig <<< "$(awk '
      /^[[:space:]]+v_mfma/                  { ++nm; mline[nm]=NR }
      /^[[:space:]]+flat_atomic_pk_add_bf16/ { ++na; aline[na]=NR }
      /^[[:space:]]+scratch_(load|store)/    { ++ns; sline[ns]=NR }
      END {
        sp=0; start=mline[1]; prev=mline[1]
        for (i=2;i<=nm;i++) { if (mline[i]-prev>2000) { sp++; lo[sp]=start; hi[sp]=prev; start=mline[i] } prev=mline[i] }
        sp++; lo[sp]=start; hi[sp]=prev
        im=0; for (j=1;j<=sp;j++) for (i=1;i<=ns;i++) if (sline[i]>=lo[j] && sline[i]<=hi[j]) im++
        h=0; for (i=1;i<=na;i++) for (j=1;j<=ns;j++) if (sline[j]<aline[i] && aline[i]-sline[j]<=40) { h++; break }
        printf "%d %d/%d", im, h, na }' "$s")"
  fi
  printf '%-6s %-8s %-9s %-9s %-9s %-8s %-8s %-10s %-14s\n' "$v" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" \
    "$(c scratch_load)" "$(c v_mfma)" "$(c flat_atomic_pk_add_bf16)" "$inm" "$sig"
done
echo "===DONE==="
