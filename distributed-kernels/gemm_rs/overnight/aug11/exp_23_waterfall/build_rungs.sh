#!/usr/bin/env bash
# exp_23: build the four waterfall rung modules from the UNMODIFIED kernel
# source. Every rung is a -D flag revert; nothing under distributed-kernels/
# is edited by this experiment.
#
# Runs inside the dhk-gemmrs container:
#   docker exec dhk-gemmrs bash <this file>
#
# Output goes to this experiment's own build/ directory, NOT harness/build --
# another agent's ablation arms live there and a collision would corrupt them.
#
# Two-step per rung, on purpose: compile once to an object with --save-temps
# (which drops the gfx942 assembly beside it), then LINK the .so from that same
# object. The disassembly then belongs to the module that will be measured
# rather than to a second, hopefully-identical compile. If the link or the
# import check fails, fall back to the one-step -shared form and record it.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
SRC=$GEMM/gemm_rs_mi300x.cpp
EXP=$GEMM/overnight/aug11/exp_23_waterfall
OUT=$EXP/build
ISA=$OUT/isa
mkdir -p "$OUT" "$ISA"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

COMMON=(
  -std=c++20 -O3
  -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math --offload-arch=gfx942
  -fPIC
  -I"$REPO/include" -I"$REPO/include/pyutils"
  -I"$ROCM_PATH/include/hip"
  -I"$PBINC" -I"$PYINC"
  -Wno-nan-infinity-disabled -ferror-limit=0
)

# name : WGM4 : RELEASE_GROUP.  FULL_ONLY is pinned to its shipped value 1 in
# every rung so it cannot act as a hidden second variable.
RUNGS=(
  "gemm_rs_w23_a:1:1"
  "gemm_rs_w23_b:0:1"
  "gemm_rs_w23_c:0:4"
  "gemm_rs_w23_null:0:4"
)

fail=0
echo "exp_23 rung build   src=$SRC"
echo "src sha256: $(sha256sum "$SRC" | cut -d' ' -f1)"
echo "hipcc: $(hipcc --version 2>&1 | head -1)"
echo

for spec in "${RUNGS[@]}"; do
  name=${spec%%:*}
  rest=${spec#*:}
  wgm=${rest%%:*}
  rg=${rest##*:}

  echo "########## $name  (WGM4=$wgm RELEASE_GROUP=$rg FULL_ONLY=1) ##########"
  tmp=$ISA/tmp_$name
  rm -rf "$tmp"; mkdir -p "$tmp"
  # A stale .so surviving a failed rebuild is exactly the masquerade this
  # experiment exists to prevent.
  rm -f "$OUT/$name.so" "$ISA/$name.s"

  DEFS=(
    -DHK_GEMM_RS_MI300X_WGM4=$wgm
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP=$rg
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=1
    -DTK_MODNAME=$name
  )

  fallback=0
  ( cd "$tmp" && hipcc "${COMMON[@]}" "${DEFS[@]}" \
      -Rpass-analysis=kernel-resource-usage \
      --save-temps -c "$SRC" -o "$tmp/$name.o" ) \
      >"$OUT/$name.compile.log" 2>&1
  cstatus=$?
  echo "  compile exit=$cstatus"
  if [ "$cstatus" != "0" ]; then
    grep -E 'error|Error' "$OUT/$name.compile.log" | head -20
  fi

  if [ -f "$tmp/$name.o" ]; then
    hipcc "${COMMON[@]}" -shared "$tmp/$name.o" -o "$OUT/$name.so" \
      >"$OUT/$name.link.log" 2>&1
    echo "  link exit=$?"
  fi

  # Fallback: the documented one-step form. Weaker (the ISA then comes from a
  # separate compile) but it keeps the ladder buildable; fingerprint.py reads
  # this marker file and discloses it.
  rm -f "$OUT/$name.fallback"
  if [ ! -f "$OUT/$name.so" ]; then
    echo "  !! two-step failed, falling back to one-step -shared build"
    hipcc "${COMMON[@]}" "${DEFS[@]}" -shared \
      -Rpass-analysis=kernel-resource-usage "$SRC" -o "$OUT/$name.so" \
      >"$OUT/$name.compile.log" 2>&1
    echo "  fallback compile exit=$?"
    fallback=1
    touch "$OUT/$name.fallback"
  fi

  # The gfx942 assembly --save-temps left behind, kept under a rung-tagged name.
  s=$(ls "$tmp"/*gfx942*.s 2>/dev/null | head -1)
  if [ -z "$s" ]; then s=$(ls "$tmp"/*amdgcn*.s 2>/dev/null | head -1); fi
  if [ -n "$s" ]; then
    cp "$s" "$ISA/$name.s"
    echo "  isa: $(wc -l < "$ISA/$name.s") lines -> $ISA/$name.s"
  else
    echo "  !! no gfx942 .s produced"
    fail=1
  fi
  # The host preprocessed output is hundreds of MB and is not evidence.
  rm -rf "$tmp"

  if [ -f "$OUT/$name.so" ]; then
    echo "  OK   $name.so ($(stat -c%s "$OUT/$name.so") bytes, fallback=$fallback)"
  else
    echo "  FAIL $name.so not produced"
    fail=1
  fi
  echo
done

echo "########## import check ##########"
for spec in "${RUNGS[@]}"; do
  name=${spec%%:*}
  [ -f "$OUT/$name.so" ] || { echo "  $name: MISSING"; fail=1; continue; }
  python3 "$EXP/import_check.py" "$OUT/$name.so" "$name" || fail=1
done

echo
[ "$fail" = "0" ] && echo "ALL RUNGS BUILT" || echo "RUNG BUILD FAILURES PRESENT"
exit $fail
