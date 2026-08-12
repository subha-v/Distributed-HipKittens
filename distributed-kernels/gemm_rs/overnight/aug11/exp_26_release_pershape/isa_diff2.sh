#!/usr/bin/env bash
# CPU-only. The resource tuple says ps2 costs nothing the incumbent does not
# already pay. This asks the stronger question: how much of the emitted code
# actually moves? A rule change confined to one scalar select should show a
# handful of instructions in the prologue and nothing anywhere else -- and if
# ps2's diff against ps0 is orders of magnitude smaller than ps1's, then the
# ~1 pp the controls moved for ps1 has a codegen explanation and ps2's does not.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
SRC=$GEMM/gemm_rs_mi300x.cpp
BASE=$GEMM/overnight/aug11/exp_26_release_pershape/isa
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBINC=$(python3 -c 'import pybind11;print(pybind11.get_include())')

emit() {
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
  local s
  s=$(ls "$d"/*gfx942*.s 2>/dev/null | head -1)
  [ -z "$s" ] && { echo "  NO ISA for $tag"; return 1; }
  cp "$s" "$BASE/$tag.s"
  echo "  $tag: $(wc -l < "$BASE/$tag.s") lines"
}

mkdir -p "$BASE"
echo "################ emitting ISA for all three rules ################"
emit ps0 0 || exit 1
emit ps1 1 || exit 1
emit ps2 2 || exit 1

echo
echo "################ diff size against the incumbent ################"
for tag in ps1 ps2; do
  diff "$BASE/ps0.s" "$BASE/$tag.s" > "$BASE/ps0_vs_$tag.diff"
  printf '  ps0 vs %-4s : %6s diff lines\n' "$tag" "$(wc -l < "$BASE/ps0_vs_$tag.diff")"
done

echo
echo "################ ps0 vs ps2, the whole diff ################"
cat "$BASE/ps0_vs_ps2.diff"

echo
echo "################ per-kernel instruction counts ################"
python3 - "$BASE/ps0.s" "$BASE/ps1.s" "$BASE/ps2.s" <<'PY'
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

a, b, c = (kernels(p) for p in sys.argv[1:4])
print(f"{'instantiation':<32}{'ps0':>8}{'ps1':>8}{'d':>6}{'ps2':>8}{'d':>6}")
print('-' * 68)
for k in sorted(set(a) | set(b) | set(c)):
    x, y, z = a.get(k, 0), b.get(k, 0), c.get(k, 0)
    print(f"{pretty(k):<32}{x:>8}{y:>8}{y-x:>+6}{z:>8}{z-x:>+6}")
PY
echo
echo "################ DONE ################"
