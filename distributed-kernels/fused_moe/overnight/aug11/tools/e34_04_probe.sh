#!/usr/bin/env bash
# exp_34 step 4: which PIECE of the M7.7 rendezvous costs the 16 B/lane?
# Build-only probes on top of B, each deleting exactly one sub-piece.
#   R1 = no publish sub-block      (barrier + poll + acquire)
#   R2 = no poll sub-block         (barrier + publish + acquire)
#   R3 = no grid_barrier           (release + publish + poll + acquire)
#   R4 = no k0p6_symmetric/peer_ptr (publish self only, 1 store)
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
FMK=$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

python3 - "$FMK" "$SC/tu" <<'PY'
import sys, pathlib, re
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])

PUB = """        if (bid == 0 && tid == 0) {
          const auto* symmetric = k0p6_symmetric(desc);
          hk_moe::release_signal_batch_system();
          for (int R = 0; R < world; ++R) {
            if (R == cur) {
              hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);
            } else {
              hk_moe::publish_epoch<hk_moe::scope::system>(
                  hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
            }
          }
        }
"""
POLL = """        if (tid < world) {
          (void)hk_moe::poll_epoch_system(
              m7_done + (size_t)tid * (size_t)T_loc_max + (size_t)T_ext,
              epoch32, (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
        }
"""
BAR = """      {
        unsigned int* gbar = (unsigned int*)k0p6_dread(desc, K0P6_D_GBAR);
        const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
        const hkp::fail_closed bar_err{pperr, 2097152};
        hkp::grid_barrier(bar, tid, bar_err);
      }
      hk_moe::acquire_payload_agent();
"""
SYMPUB = """          const auto* symmetric = k0p6_symmetric(desc);
          hk_moe::release_signal_batch_system();
          for (int R = 0; R < world; ++R) {
            if (R == cur) {
              hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);
            } else {
              hk_moe::publish_epoch<hk_moe::scope::system>(
                  hk_moe::peer_ptr(self_slot, R, symmetric), epoch32);
            }
          }
"""
for name, needle in (("PUB", PUB), ("POLL", POLL), ("BAR", BAR), ("SYMPUB", SYMPUB)):
    assert src.count(needle) == 1, (name, src.count(needle))

variants = {
    "R1": src.replace(PUB, ""),
    "R2": src.replace(POLL, ""),
    "R3": src.replace(BAR, ""),
    "R4": src.replace(SYMPUB,
        "          hk_moe::release_signal_batch_system();\n"
        "          hk_moe::publish_epoch<hk_moe::scope::agent>(self_slot, epoch32);\n"),
}
for name, text in variants.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "k0pf6gm_device_tile_mps.hip").write_text(text)
    print(name, "written", len(text))
PY

docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla; SC=$H/e34; O=$SC/out
DHK=$SC/DHK; FM=$DHK/distributed-kernels/fused_moe
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
COMMON=(--genco --offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in R1 R2 R3 R4; do
  hipcc "${COMMON[@]}" $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco \
    > $O/$v.log 2>&1
  echo "$v exit=$? errors=$(grep -cE "error:" $O/$v.log)"
done
'
echo
printf '%-4s %-8s %-6s %-6s %-8s %-10s %-10s %-8s\n' arm SGPR VGPR AGPR scratch sgpr_spill vgpr_spill LDS
for v in A B R1 R2 R3 R4; do
  f=$O/$v.log
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  printf '%-4s %-8s %-6s %-6s %-8s %-10s %-10s %-8s\n' "$v" \
    "$(g 'TotalSGPRs')" "$(g '^.*VGPRs:')" "$(g 'AGPRs:')" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" "$(g 'LDS Size')"
done
echo "===DONE==="
