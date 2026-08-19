#!/bin/bash
# ---------------------------------------------------------------------------
# G-L0b build gate for the boundary rig: .text identity + resource receipts.
#
# Asserts the two halves of the plan's build-identity rule (UNDERSTANDING_PLAN
# §4b): with M25_LEDGER=0 the emitted device .text must be BYTE-IDENTICAL to
# the pre-instrument build, and with M25_LEDGER=1 it must differ. Also prints
# the per-kernel resource tuple (VGPR/AGPR/SGPR/LDS/spills/scratch) for every
# build so the ledger's spill delta is a receipt, not a claim.
#
# Usage:  ./ledger_build_gate.sh <base_src_dir> <new_src_dir> [outdir]
#   <base_src_dir> / <new_src_dir> each hold m25_boundary_bench.hip and
#   m25_cdar.cuh; the repo root (containing include/) is taken as
#   <dir>/../..  exactly as the normal build does.
# ---------------------------------------------------------------------------
set -u
export PATH=/opt/rocm/llvm/bin:$PATH   # llvm-objcopy / clang-offload-bundler
BASE=${1:?base src dir}
NEW=${2:?new src dir}
OUT=${3:-$PWD/ledger_gate_out}
ARCH=${ARCH:-gfx950}
mkdir -p "$OUT"

co() {  # $1 = src dir, $2 = tag, $3.. = extra defines
    local dir=$1 tag=$2; shift 2
    local root
    root=$(cd "$dir/../.." && pwd)
    hipcc --offload-arch="$ARCH" -std=c++20 -O3 --cuda-device-only \
        -I "$root" "$@" -o "$OUT/$tag.co" "$dir/m25_boundary_bench.hip" \
        2> "$OUT/$tag.build.log"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "BUILD FAIL $tag (see $OUT/$tag.build.log)"; tail -30 "$OUT/$tag.build.log"; return 1
    fi
    # --cuda-device-only emits a clang offload BUNDLE; pull the gfx ELF out.
    if head -c 24 "$OUT/$tag.co" | grep -q "__CLANG_OFFLOAD_BUNDLE__"; then
        clang-offload-bundler --type=o --unbundle \
            --targets=hipv4-amdgcn-amd-amdhsa--"$ARCH" \
            --input="$OUT/$tag.co" --output="$OUT/$tag.elf" \
            2>> "$OUT/$tag.build.log" || cp "$OUT/$tag.co" "$OUT/$tag.elf"
    else
        cp "$OUT/$tag.co" "$OUT/$tag.elf"
    fi
    llvm-objcopy --dump-section=.text="$OUT/$tag.text" "$OUT/$tag.elf" \
        /dev/null 2>> "$OUT/$tag.build.log"
    if [ ! -s "$OUT/$tag.text" ]; then
        echo "TEXT EXTRACT FAIL $tag"; return 1
    fi
    printf '%s  %s  (.text %s bytes)\n' \
        "$(sha256sum "$OUT/$tag.text" | cut -d' ' -f1)" "$tag" \
        "$(stat -c%s "$OUT/$tag.text")"
}

resources() {  # $1 = src dir, $2 = tag, $3.. = extra defines
    local dir=$1 tag=$2; shift 2
    local root
    root=$(cd "$dir/../.." && pwd)
    hipcc --offload-arch="$ARCH" -std=c++20 -O3 --cuda-device-only -c \
        -Rpass-analysis=kernel-resource-usage -I "$root" "$@" \
        -o /dev/null "$dir/m25_boundary_bench.hip" \
        > "$OUT/$tag.res.log" 2>&1
    echo "--- resources: $tag"
    grep -E "Function Name|SGPRs|VGPRs|AGPRs|ScratchSize|Occupancy|Spills|LDS" \
        "$OUT/$tag.res.log" | sed 's/^.*remark: //'
}

echo "=== .text sha256 (arch=$ARCH) ==="
co "$BASE" base_off               || exit 1
co "$NEW"  new_off                || exit 1
co "$NEW"  new_on   -DM25_LEDGER=1 || exit 1
co "$NEW"  new_nocompute -DM25_NOCOMPUTE=1 || exit 1

echo
B=$(sha256sum "$OUT/base_off.text" | cut -d' ' -f1)
N=$(sha256sum "$OUT/new_off.text"  | cut -d' ' -f1)
L=$(sha256sum "$OUT/new_on.text"   | cut -d' ' -f1)
if [ "$B" = "$N" ]; then echo "GATE_TEXT_PARITY PASS  (M25_LEDGER=0 .text == pre-instrument .text)";
else echo "GATE_TEXT_PARITY FAIL  base=$B new=$N"; fi
if [ "$N" != "$L" ]; then echo "GATE_TEXT_DIFFERS PASS (M25_LEDGER=1 .text != M25_LEDGER=0 .text)";
else echo "GATE_TEXT_DIFFERS FAIL (ledger build is identical — instrument not compiled in)"; fi

echo
echo "=== resource tuples ==="
resources "$BASE" base_off
resources "$NEW"  new_off
resources "$NEW"  new_on -DM25_LEDGER=1
echo
echo "outputs in $OUT"
