#!/usr/bin/env bash
# exp_34 step 6: probes on the MERGED rendezvous, to close the last 16 B/lane.
#   N1 = readfirstlane on `coarse75` (n2_phase2_gm_mps.cpp:380's documented fix:
#        a descriptor-derived bool cannot be proven uniform, so LLVM parks it in a
#        VGPR and any branch on it extends a VECTOR live range)
#   N2 = N1 + Ready instantiation reverted to mode 12's (build probe only)
#   N3 = merged with the coarse PUBLISH arm removed  (attribution)
#   N4 = merged with the coarse POLL+acquire block removed (attribution)
# Also: correct ISA extraction with --cuda-device-only -S (llvm-objdump cannot
# read the --genco offload bundle; the earlier 1-line .isa files were the error).
# NO GPU WORK.
set -uo pipefail
H=$HOME; SC=$H/e34; O=$SC/out
FMK=$SC/DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip

python3 - "$FMK" "$SC/tu" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])

COARSE = "    const bool coarse75 = hk_moe::mps::mode_is_coarse(cfg75);\n"
RFL = ("    const bool coarse75 = __builtin_amdgcn_readfirstlane(\n"
       "        (int)hk_moe::mps::mode_is_coarse(cfg75)) != 0;\n")
READY_INST = "false, true, true>"
PUB = """      if (coarse75) {
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
"""
POLLBLK = """    if (coarse75) {
      if (payload_ok && tid < world) {
        (void)hk_moe::poll_epoch_system(
            row_ready + (size_t)tid * T_loc_max + (size_t)T_ext, epoch32,
            (std::uint64_t)spin_limit,
            hk_moe::set_error_bit{pperr, K0P6_MPS_ERR_M7DONE});
      }
      __syncthreads();
      hk_moe::acquire_payload_system();
    }
"""
for nm, nd in (("COARSE", COARSE), ("READY", READY_INST), ("PUB", PUB), ("POLL", POLLBLK)):
    assert src.count(nd) == 1, (nm, src.count(nd))

n1 = src.replace(COARSE, RFL)
variants = {
    "N1": n1,
    "N2": n1.replace(READY_INST, "false, true>"),
    "N3": src.replace(PUB, "      {\n"),
    "N4": src.replace(POLLBLK, ""),
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
COMMON=(--offload-arch=gfx950 -std=c++20 -O3
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
  -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3
  -Rpass-analysis=kernel-resource-usage
  -I$DHK/include -I$FM
  -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip
  -I$MR -I$MR/include -I$MR/src
  -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include)
for v in N1 N2 N3 N4; do
  hipcc "${COMMON[@]}" --genco $SC/tu/$v/k0pf6gm_device_tile_mps.hip -o $O/$v.hsaco > $O/$v.log 2>&1
  echo "$v genco exit=$? errors=$(grep -cE "error:" $O/$v.log)"
done
# ---- correct ISA extraction: device-only -S emits gfx950 assembly text ----
for v in A B N1; do
  case $v in
    A) SRC=$SC/base_tu.hip ;;
    B) SRC=$FM/k0pf6gm_device_tile_mps.hip ;;
    N1) SRC=$SC/tu/N1/k0pf6gm_device_tile_mps.hip ;;
  esac
  if [ "$v" = A ]; then
    mkdir -p $SC/atu && cp $SC/base/k0pf6gm_device_tile_mps.hip.orig $SC/atu/k0pf6gm_device_tile_mps.hip
    cp $SC/base/moe_mps_adapter.cuh.orig $SC/atu/moe_mps_adapter.cuh
    SRC=$SC/atu/k0pf6gm_device_tile_mps.hip
    hipcc "${COMMON[@]}" -I$SC/atu --cuda-device-only -S $SRC -o $O/$v.s > $O/$v.slog 2>&1
  else
    hipcc "${COMMON[@]}" --cuda-device-only -S $SRC -o $O/$v.s > $O/$v.slog 2>&1
  fi
  echo "$v -S exit=$? lines=$(wc -l < $O/$v.s 2>/dev/null)"
done
'
echo
echo "############ TUPLES ############"
printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s\n' arm SGPR VGPR AGPR scratch sgpr_spill vgpr_spill LDS
for v in A B N1 N2 N3 N4; do
  f=$O/$v.log
  g() { grep -m1 "$1" "$f" | sed 's|.*: *||;s| \[.*||'; }
  printf '%-4s %-6s %-6s %-6s %-8s %-11s %-11s %-8s\n' "$v" \
    "$(g 'TotalSGPRs')" "$(g '^.*VGPRs:')" "$(g 'AGPRs:')" \
    "$(g 'ScratchSize')" "$(g 'SGPRs Spill')" "$(g 'VGPRs Spill')" "$(g 'LDS Size')"
done
echo
echo "############ ASSEMBLY CENSUS (from -S) ############"
printf '%-4s %-10s %-8s %-13s %-9s %-9s\n' arm lines v_mfma pk_add_bf16 scr_load scr_store
for v in A B N1; do
  [ -s "$O/$v.s" ] || { echo "$v: NO .s"; continue; }
  printf '%-4s %-10s %-8s %-13s %-9s %-9s\n' "$v" \
    "$(wc -l < $O/$v.s)" \
    "$(grep -cE '^[[:space:]]+v_mfma' $O/$v.s)" \
    "$(grep -cE '^[[:space:]]+flat_atomic_pk_add_bf16' $O/$v.s)" \
    "$(grep -cE '^[[:space:]]+scratch_load' $O/$v.s)" \
    "$(grep -cE '^[[:space:]]+scratch_store' $O/$v.s)"
done
echo "===DONE==="
