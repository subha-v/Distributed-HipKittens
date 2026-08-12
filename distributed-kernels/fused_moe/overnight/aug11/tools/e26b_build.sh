#!/usr/bin/env bash
# exp_26 follow-up step 1: seven compile-only builds from the SAME immutable
# snapshot as the first run -- D (donor include) plus the vendored file at
# N2GM_P1_SCHED_GSCALE in {0,2,4,6,9,15}. NO GPU WORK.
set -uo pipefail
SC=$HOME/overnight-scratch/e26
DHK=$SC/dhk
FM=$DHK/distributed-kernels/fused_moe
K0=$HOME/amd-master/auto-gpu-kernel/k0_fused_moe
mkdir -p "$SC/out2"

echo "===INSTALL_VENDORED==="
tr -d '\r' < "$SC/incoming/n2_phase1_gm_mps.cpp" > "$FM/n2_phase1_gm_mps.cpp"
sha256sum "$FM/n2_phase1_gm_mps.cpp"
wc -l "$FM/n2_phase1_gm_mps.cpp" "$K0/solution/hip/n2_phase1_gm.cpp"

echo "===VENDOR_DIFF_VS_DONOR==="
diff -u "$K0/solution/hip/n2_phase1_gm.cpp" "$FM/n2_phase1_gm_mps.cpp" > "$SC/out2/vendor.diff"
ins=$(grep -c '^+[^+]' "$SC/out2/vendor.diff")
del=$(grep -c '^-[^-]' "$SC/out2/vendor.diff")
echo "diffstat: +${ins} -${del}"
echo "--- the deleted donor lines (must all reappear verbatim in the mask table) ---"
grep '^-[^-]' "$SC/out2/vendor.diff"

echo "===TU_SANITY (reusing the first run's throwaway TUs)==="
diff "$SC/tu/b0/k0pf6gm_device_tile_mps.hip" "$SC/tu/b1/k0pf6gm_device_tile_mps.hip"
echo "(one-line include difference above)"

echo "===BUILD (subha_k1, CPU-only genco, 4-way parallel)==="
docker exec subha_k1 bash -lc '
set -uo pipefail
H=/home/subvadla
SC=$H/overnight-scratch/e26
DHK=$SC/dhk
K0=$H/amd-master/auto-gpu-kernel/k0_fused_moe
MR=/usr/local/lib/python3.12/dist-packages/mori/_jit-sources
build_one() {
  v=$1
  case $v in
    D) SRC=$SC/tu/b0/k0pf6gm_device_tile_mps.hip; EXTRA="" ;;
    *) SRC=$SC/tu/b1/k0pf6gm_device_tile_mps.hip; EXTRA="-DN2GM_P1_SCHED_GSCALE=${v#M}" ;;
  esac
  t0=$SECONDS
  hipcc --genco --offload-arch=gfx950 -std=c++20 -O3 \
    -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -mllvm -amdgpu-mfma-vgpr-form=1 -DK0P6GM_G=3 -DN2GM_G=3 \
    -Rpass-analysis=kernel-resource-usage \
    -I$DHK/include -I$DHK/distributed-kernels/fused_moe \
    -I$K0/solution/hip/hkp -I$K0/prefill_opt/kernels -I$K0/solution/hip \
    -I$MR -I$MR/include -I$MR/src \
    -I$MR/3rdparty/spdlog/include -I$MR/3rdparty/msgpack-c/include \
    $EXTRA "$SRC" -o $SC/out2/$v.hsaco > $SC/out2/$v.log 2>&1
  rc=$?
  echo "$v exit=$rc secs=$((SECONDS-t0)) errors=$(grep -cE "error:" $SC/out2/$v.log)"
}
export -f build_one; export SC DHK K0 MR
printf "%s\n" D M0 M2 M4 M6 M9 M15 | xargs -P 4 -I{} bash -c "build_one {}"
' 2>&1 | tail -20

echo "===HASHES==="
sha256sum "$SC/out2/"*.hsaco 2>/dev/null
