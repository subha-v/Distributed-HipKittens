#!/usr/bin/env bash
# it02: reducer-count (NR) response on the two hottest shapes, official
# evaluator only. Builds gemm_rs_mi300x_nrenv.so from the patched source
# (HK_GEMM_RS_NR resolve-time override; production .so untouched), then one
# official benchmark pass per setting. Table NR: row5=32, row6=48.
set -uo pipefail

REPO=/home/subvadla/dhk
GEMM=$REPO/distributed-kernels/gemm_rs
ON=$GEMM/overnight
T=$ON/tools
OURS=$ON/compbench/ours
IT=$ON/aug13/iter_loop
OUT=$ON/harness/build
NAME=dhk-gemmrs

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
. "$T/kfd_live.sh"
mkdir -p "$IT/it02"

bash "$T/gpu_lease.sh" acquire iter_it02 5400 || { LOG "no lease"; exit 1; }
trap 'bash "$T/gpu_lease.sh" release iter_it02' EXIT
bash "$T/set_clocks.sh" pin 1900 >/dev/null

LOG "build gemm_rs_mi300x_nrenv.so"
docker exec "$NAME" bash -c "
  PYINC=\$(python3 -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])')
  PBINC=\$(python3 -c 'import pybind11;print(pybind11.get_include())')
  hipcc -std=c++20 -O3 -DKITTENS_CDNA3 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
    -ffast-math --offload-arch=gfx942 -shared -fPIC \
    -DTK_MODNAME=gemm_rs_mi300x_nrenv \
    -I$REPO/include -I$REPO/include/pyutils -I/opt/rocm/include/hip \
    -I\$PBINC -I\$PYINC -Wno-nan-infinity-disabled -ferror-limit=0 \
    $GEMM/gemm_rs_mi300x.cpp -o $OUT/gemm_rs_mi300x_nrenv.so" \
    > "$IT/it02/build.log" 2>&1
[ -f "$OUT/gemm_rs_mi300x_nrenv.so" ] || { LOG "BUILD FAILED"; tail -20 "$IT/it02/build.log"; exit 1; }
LOG "build ok ($(stat -c%s $OUT/gemm_rs_mi300x_nrenv.so) bytes)"

pass() { # pass <label> <nr_spec or ->
    local label=$1 spec=$2 envfrag=""
    [ "$spec" != "-" ] && envfrag="HK_GEMM_RS_NR=$spec"
    kfd_wait_clean 30 10 || { LOG "node not clean before $label"; exit 1; }
    LOG "official benchmark: $label (NR='${spec}')"
    docker exec "$NAME" bash -c "cd $OURS && POPCORN_FD=3 POPCORN_GPUS=8 \
        HK_BUILD_DIR=$ON/harness/build HK_DEBUG=0 \
        HK_KERNEL_MODULE=gemm_rs_mi300x_nrenv $envfrag \
        TMPDIR=$OURS/.tmp TORCH_EXTENSIONS_DIR=$OURS/.ext \
        timeout 1500 python3 eval.py benchmark cases_bench.txt \
        3>$OURS/benchmark.popcorn.txt >$OURS/benchmark.stdout.txt 2>$OURS/benchmark.stderr.txt"
    cp "$OURS/benchmark.popcorn.txt" "$IT/it02/$label.popcorn.txt"
    grep -q "^check: pass" "$IT/it02/$label.popcorn.txt" \
        || { LOG "$label: OFFICIAL CHECK FAIL"; exit 1; }
    LOG "$label: check pass"
}

pass base -
pass nrA "5:24,6:32"
pass nrB "5:40,6:40"
pass nrC "5:48,6:56"

LOG "comparison (it02, deltas vs base, shapes 5/6 are the targets)"
python3 - "$IT/it02" <<'EOF'
import glob, math, re, sys
def parse(p):
    f = {}
    for line in open(p):
        m = re.match(r"benchmark\.(\d+)\.best: (.*)", line)
        if m:
            f[int(m.group(1))] = float(m.group(2)) / 1000.0
    return [f[i] for i in range(6)]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
d = sys.argv[1]
base = parse(f"{d}/base.popcorn.txt")
print(f"{'arm':>5} {'8192a':>8} {'8192b':>8} {'GM':>8}   deltas vs base")
print(f"{'base':>5} {base[4]:8.1f} {base[5]:8.1f} {gm(base):8.1f}")
for p in sorted(glob.glob(f"{d}/nr*.popcorn.txt")):
    c = parse(p)
    tag = p.split("/")[-1].split(".")[0]
    print(f"{tag:>5} {c[4]:8.1f} {c[5]:8.1f} {gm(c):8.1f}   "
          f"s5 {100*(c[4]-base[4])/base[4]:+.2f}%  "
          f"s6 {100*(c[5]-base[5])/base[5]:+.2f}%  "
          f"GM {100*(gm(c)-gm(base))/gm(base):+.2f}%")
EOF
LOG "ITER DONE"
