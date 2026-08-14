#!/usr/bin/env bash
# Lean iteration driver: one official-evaluator benchmark pass per module,
# back-to-back in one lease, then a per-shape comparison. Node HOST.
#
# Usage: iter_bench.sh <iter_tag> <module_base> <module_cand>
#   e.g. iter_bench.sh it01 gemm_rs_mi300x gemm_rs_mi300x_cmid
#
# Official setup only: unmodified eval.py, benchmark mode (its checked first
# call per case is the correctness gate), one process per rank, official
# shapes/seeds. Hygiene kept: lease, pinned clocks, HK_DEBUG=0 asserted.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
T=$ON/tools
OURS=$ON/compbench/ours
IT=$ON/aug13/iter_loop
NAME=dhk-gemmrs
TAG=${1:?tag}; BASE=${2:?base module}; CAND=${3:?candidate module}

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
. "$T/kfd_live.sh"
mkdir -p "$IT/$TAG"

bash "$T/gpu_lease.sh" acquire "iter_$TAG" 3600 || { LOG "no lease"; exit 1; }
trap 'bash "$T/gpu_lease.sh" release "iter_'"$TAG"'"' EXIT
bash "$T/set_clocks.sh" pin 1900 >/dev/null

pass() { # pass <label> <module>
    local label=$1 mod=$2
    kfd_wait_clean 30 10 || { LOG "node not clean before $label"; exit 1; }
    LOG "official benchmark: $label ($mod)"
    docker exec "$NAME" bash -c "cd $OURS && POPCORN_FD=3 POPCORN_GPUS=8 \
        HK_BUILD_DIR=$ON/harness/build HK_DEBUG=0 HK_KERNEL_MODULE=$mod \
        TMPDIR=$OURS/.tmp TORCH_EXTENSIONS_DIR=$OURS/.ext \
        timeout 1500 python3 eval.py benchmark cases_bench.txt \
        3>$OURS/benchmark.popcorn.txt >$OURS/benchmark.stdout.txt 2>$OURS/benchmark.stderr.txt"
    cp "$OURS/benchmark.popcorn.txt" "$IT/$TAG/$label.popcorn.txt"
    grep -q "^check: pass" "$IT/$TAG/$label.popcorn.txt" \
        || { LOG "$label: OFFICIAL CHECK FAIL"; exit 1; }
    local n
    n=$(grep -c '\[hk ' "$OURS/benchmark.stderr.txt" 2>/dev/null || true)
    [ "${n:-0}" -eq 0 ] || { LOG "$label: HK_DEBUG leak ($n lines)"; exit 1; }
    LOG "$label: check pass"
}

pass base "$BASE"
pass cand "$CAND"

LOG "comparison ($TAG)"
python3 - "$IT/$TAG" <<'EOF'
import math, re, sys
def parse(p):
    f = {}
    for line in open(p):
        m = re.match(r"benchmark\.(\d+)\.best: (.*)", line)
        if m:
            f[int(m.group(1))] = float(m.group(2)) / 1000.0
    return [f[i] for i in range(6)]
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
d = sys.argv[1]
b, c = parse(f"{d}/base.popcorn.txt"), parse(f"{d}/cand.popcorn.txt")
shapes = ["64", "512", "2048", "4096", "8192a", "8192b"]
print(f"{'shape':>6} {'base':>8} {'cand':>8} {'delta':>8}")
for s, x, y in zip(shapes, b, c):
    print(f"{s:>6} {x:8.1f} {y:8.1f} {100*(y-x)/x:+7.2f}%")
print(f"{'GM':>6} {gm(b):8.1f} {gm(c):8.1f} {100*(gm(c)-gm(b))/gm(b):+7.2f}%")
EOF
LOG "ITER DONE"
