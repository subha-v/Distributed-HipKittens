#!/usr/bin/env bash
# exp_34 step 7: find a shape for stage (4)+(5) that keeps the ratchet's tuple.
#   N5 = drop the HOISTED acquire; keep poll + __syncthreads and let M8 keep its
#        per-batch acquire (so `Ready` compiles out only the poll and the ballot).
#        The __syncthreads is NOT optional: without it a thread that never polled
#        could read slots before another thread's poll returned.
#   N6 = N5 + the poll moved to tid == 0 with an 8-iteration loop (one thread's
#        live values instead of eight threads' divergent ones).
#   N7 = N5 + readfirstlane on coarse75.
#   N8 = N6 + readfirstlane on coarse75.
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
FMK=$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

python3 - "$FMK" "$SC/tu" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])

ACQ_HOISTED = """      __syncthreads();
      hk_moe::acquire_payload_system();
    }
"""
ACQ_DROPPED = """      __syncthreads();
    }
"""
M8_READY = """  if constexpr (!Ready) {
    if (__ballot(batch_ready) != ~0ull) return;
    // Only lanes 0..31 poll readiness, while all lanes consume the shuffled
    // bases. Apply one convergent wave acquire before any payload load.
    hk_moe::acquire_payload_system();
  } else {
    (void)batch_ready;
  }
"""
M8_KEEPACQ = """  if constexpr (!Ready) {
    if (__ballot(batch_ready) != ~0ull) return;
  } else {
    (void)batch_ready;
  }
  // Only lanes 0..31 poll readiness (when they poll at all), while all lanes
  // consume the shuffled bases. One convergent wave acquire before any load.
  hk_moe::acquire_payload_system();
"""
POLL8 = """      if (payload_ok && tid < world) {
        (void)hk_moe::poll_epoch_system(
            row_ready + (size_t)tid * T_loc_max + (size_t)T_ext, epoch32,
            (std::uint64_t)spin_limit,
            hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
      }
"""
POLL1 = """      if (payload_ok && tid == 0) {
        for (int p = 0; p < world; ++p) {
          (void)hk_moe::poll_epoch_system(
              row_ready + (size_t)p * T_loc_max + (size_t)T_ext, epoch32,
              (std::uint64_t)spin_limit,
              hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
        }
      }
"""
COARSE = "    const bool coarse75 = hk_moe::mps::mode_is_coarse(cfg75);\n"
RFL = ("    const bool coarse75 = __builtin_amdgcn_readfirstlane(\n"
       "        (int)hk_moe::mps::mode_is_coarse(cfg75)) != 0;\n")
for nm, nd in (("ACQ", ACQ_HOISTED), ("M8", M8_READY), ("POLL8", POLL8), ("COARSE", COARSE)):
    assert src.count(nd) == 1, (nm, src.count(nd))

n5 = src.replace(ACQ_HOISTED, ACQ_DROPPED).replace(M8_READY, M8_KEEPACQ)
n6 = n5.replace(POLL8, POLL1)
variants = {"N5": n5, "N6": n6,
            "N7": n5.replace(COARSE, RFL), "N8": n6.replace(COARSE, RFL)}
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
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in N5 N6 N7 N8; do
  hipcc "${COMMON[@]}" --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  echo "$v exit=$? errors=$(grep -cE "error:" $O/$v.log)"
  hipcc "${COMMON[@]}" --cuda-device-only -S $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.s > /dev/null 2>&1
done
# N4 assembly too (it is the 128 B reference point)
hipcc "${COMMON[@]}" --cuda-device-only -S $SC/tu/N4/k0pf6gm_device_tile_mps.hip -o $O/N4.s > /dev/null 2>&1
echo asm_done'

echo
printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s %-9s %-9s %-7s %-7s\n' \
  arm SGPR VGPR AGPR scratch sgpr_spill vgpr_spill LDS scr_load scr_store mfma pk_add
for v in A B N4 N5 N6 N7 N8; do
  f=$O/$v.log
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  s=$O/$v.s
  c() { [ -s "$s" ] && grep -cE "^[[:space:]]+$1" "$s" || echo "-"; }
  printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s %-9s %-9s %-7s %-7s\n' "$v" \
    "$(g 'TotalSGPRs')" "$(g '^.*VGPRs:')" "$(g 'AGPRs:')" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" "$(g 'LDS Size')" \
    "$(c scratch_load)" "$(c scratch_store)" "$(c v_mfma)" "$(c flat_atomic_pk_add_bf16)"
done
echo "===DONE==="
