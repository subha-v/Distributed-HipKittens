#!/usr/bin/env bash
# CPU-only. Prove the PERSHAPE define actually reached codegen, and price its
# collateral on the instantiations it must not disturb.
#
# Why this exists: push.ps1 rewrites every source mtime, so mtime is never a
# freshness test on this node, and "the build succeeded" says nothing about
# whether -D landed (the source guards every macro with #ifndef, so a typo in
# the flag name silently compiles the default). An ISA diff is the cheapest
# check that is actually behavioural: if PERSHAPE=1 changed nothing in the
# emitted code, the candidate is the incumbent under another name.
#
# It also bounds the codegen risk to the four control shapes. rows 1/2/3 and the
# generic row are separate instantiations from rows 4/5/6, and only the
# rgroup expression differs between the two builds, so any instruction-count
# delta outside the scalar prologue would be a red flag the timing run could not
# distinguish from a release effect.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
SRC=$GEMM/gemm_rs_mi300x.cpp
BASE=$GEMM/overnight/aug11/exp_26_release_pershape/isa
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

emit() {                      # emit <tag> <pershape>
  # Separate statements on purpose: `local a=$1 b=$BASE/$a` expands every word
  # before the builtin assigns any of them, so `$a` is unbound under `set -u`.
  local tag=$1
  local ps=$2
  local d=$BASE/$tag
  rm -rf "$d"; mkdir -p "$d"; cd "$d"
  hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
    -ffast-math --offload-arch=gfx942 -DTK_MODNAME=gemm_rs_mi300x \
    -I"$REPO/include" -I"$REPO/include/pyutils" -I"$GEMM" \
    -I"$ROCM_PATH/include/hip" -I"$PBINC" -I"$PYINC" \
    -Wno-nan-infinity-disabled \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP=4 \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_FULL_ONLY=1 \
    -DHK_GEMM_RS_MI300X_RELEASE_GROUP_PERSHAPE=$ps \
    --save-temps -c "$SRC" -o "$d/x.o" >"$d/build.log" 2>&1
  echo "  $tag (PERSHAPE=$ps) exit=$?"
  local s
  s=$(ls "$d"/*gfx942*.s 2>/dev/null | head -1)
  [ -z "$s" ] && { echo "  NO ISA for $tag"; return 1; }
  cp "$s" "$BASE/$tag.s"
  wc -l "$BASE/$tag.s"
}

mkdir -p "$BASE"
echo "################ emitting ISA for both rules ################"
emit ps0 0 || exit 1
emit ps1 1 || exit 1

echo
echo "################ the diff must be NON-EMPTY (the flag landed) ################"
diff "$BASE/ps0.s" "$BASE/ps1.s" > "$BASE/ps0_vs_ps1.diff"
n=$(wc -l < "$BASE/ps0_vs_ps1.diff")
echo "diff lines: $n"
if [ "$n" = "0" ]; then
  echo "VERDICT: IDENTICAL ISA -- the PERSHAPE define did NOT reach codegen."
  exit 1
fi
head -60 "$BASE/ps0_vs_ps1.diff"

echo
echo "################ per-kernel instruction counts, both builds ################"
# Split the .s by kernel symbol and count instructions per instantiation, so a
# change confined to one instantiation's scalar prologue is distinguishable from
# a schedule change in the mainloop of a control instantiation.
python3 - "$BASE/ps0.s" "$BASE/ps1.s" <<'PY'
import re, sys

def kernels(path):
    out, cur, count = {}, None, 0
    for line in open(path, errors='replace'):
        m = re.match(r'^([_A-Za-z][\w.$]*):\s*(;.*)?$', line)
        if m and 'gemm_rs_mi300x_kernel' in m.group(1):
            if cur: out[cur] = count
            cur, count = m.group(1), 0
            continue
        if cur is not None:
            if line.startswith('.Lfunc_end'):
                out[cur] = count; cur = None; continue
            s = line.strip()
            if s and not s.startswith(('.', ';', '//')):
                count += 1
    if cur: out[cur] = count
    return out

def pretty(name):
    nums = re.findall(r'Li(\d+)E', name)
    tail = re.search(r'Lb(\d)E', name)
    return (f"BM={nums[0]:>3} BN={nums[1]:>3} BK={nums[2]:>3} tail={tail.group(1)}"
            if len(nums) >= 3 and tail else name[:40])

a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
print(f"{'instantiation':<32}{'ps0':>8}{'ps1':>8}{'delta':>8}")
print('-' * 56)
for k in sorted(set(a) | set(b)):
    x, y = a.get(k, 0), b.get(k, 0)
    print(f"{pretty(k):<32}{x:>8}{y:>8}{y-x:>+8}")
PY

echo
echo "################ ordering-op counts (must not move) ################"
for tag in ps0 ps1; do
  printf '%-6s' "$tag"
  for pat in buffer_wbl2 buffer_inv s_barrier 'v_mfma' scratch_store scratch_load; do
    printf ' %s=%s' "$pat" "$(grep -cE "$pat" "$BASE/$tag.s")"
  done
  echo
done

echo
echo "################ DONE ################"
