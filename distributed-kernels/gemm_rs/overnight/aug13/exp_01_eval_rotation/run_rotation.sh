#!/usr/bin/env bash
# exp_01 (aug13): instrument-B (official evaluator) ladder with arm-order
# rotation and an ours-debug arm. See plan.md. Runs on the node HOST.
#
# Arms: O=ours(HK_DEBUG=0)  D=ours_dbg(HK_DEBUG=1)  R=reference  K=rank1.
# Staging mirrors tools/run_ours_evaluator.sh, tools/run_reference_arm.sh and
# tools/run_rank1_bench3.sh (the repaired 2026-08-12 driver) verbatim — those
# scripts re-stage (rm -rf) their arm directory on every invocation, so the
# bench-only session passes below carry each script's env block instead of
# calling the script itself.
set -uo pipefail

ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
T=$ON/tools
EXP=$ON/aug13/exp_01_eval_rotation
RAW=$EXP/raw
NAME=dhk-gemmrs
SRC=/home/subvadla/ddt-exp026-o1-stock-gemm-rs-test-de730f29/runtime/exp026-stock-gemm-rs-test-de730f29/stock_gemm_rs__test/cwd
RANK1SUB=/home/subvadla/amd-master/auto-gpu-kernel/k2_mi300x_megakernel/references/submissions/gemm_rs_rank1_58abcf.py
COMPAT=$ON/tools/compat
IRISDST=/usr/local/lib/python3.10/dist-packages
OURS=$ON/compbench/ours
REF=$ON/compbench/reference
R1=$ON/compbench/rank1

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
run() { docker exec "$NAME" bash -c "$1"; }
die() { LOG "ABORT: $*"; exit 1; }

. "$T/kfd_live.sh"
mkdir -p "$RAW"

CASES_BENCH='world_size: 8; m: 64; n: 7168; k: 18432; has_bias: False; seed: 1234
world_size: 8; m: 512; n: 4096; k: 12288; has_bias: True; seed: 663
world_size: 8; m: 2048; n: 2880; k: 2880; has_bias: True; seed: 166
world_size: 8; m: 4096; n: 4096; k: 4096; has_bias: False; seed: 1371
world_size: 8; m: 8192; n: 4096; k: 14336; has_bias: True; seed: 7168
world_size: 8; m: 8192; n: 8192; k: 29568; has_bias: False; seed: 42'

# ---------------------------------------------------------------- utilities
save() { # save <arm_dir> <label> <dest>  (popcorn + stderr tail; gz full stderr)
    local dir=$1 label=$2 dest=$3
    mkdir -p "$dest"
    cp "$dir/$label.popcorn.txt" "$dest/" 2>/dev/null || true
    tail -60 "$dir/$label.stderr.txt" > "$dest/$label.stderr.tail" 2>/dev/null || true
    gzip -c "$dir/$label.stderr.txt" > "$dest/$label.stderr.txt.gz" 2>/dev/null || true
}

check_pass() { # check_pass <popcorn_file> <what>
    grep -q "^check: pass" "$1" || die "$2 did not pass ($1)"
    LOG "$2: check pass"
}

ours_env() { # ours_env <dbg>
    echo "POPCORN_FD=3 POPCORN_GPUS=8 HK_BUILD_DIR=$ON/harness/build HK_DEBUG=$1 TMPDIR=$OURS/.tmp TORCH_EXTENSIONS_DIR=$OURS/.ext"
}

bench_ours() { # bench_ours <dbg 0|1> <dest>
    local dbg=$1 dest=$2
    kfd_wait_clean 30 10 || die "node not clean before ours(dbg=$dbg)"
    LOG "bench ours dbg=$dbg -> $dest"
    run "cd $OURS && $(ours_env "$dbg") timeout 1500 python3 eval.py benchmark cases_bench.txt 3>$OURS/benchmark.popcorn.txt >$OURS/benchmark.stdout.txt 2>$OURS/benchmark.stderr.txt"
    save "$OURS" benchmark "$dest"
    check_pass "$dest/benchmark.popcorn.txt" "ours(dbg=$dbg) bench"
    if [ "$dbg" = "0" ]; then
        local n
        n=$(grep -c '\[hk ' "$OURS/benchmark.stderr.txt" 2>/dev/null || true)
        [ "${n:-0}" -eq 0 ] || die "fast path not taken: $n [hk ] lines in ours stderr"
        LOG "fast path verified: 0 debug lines"
    fi
}

bench_reference() { # bench_reference <dest>
    local dest=$1
    kfd_wait_clean 30 10 || die "node not clean before reference"
    LOG "bench reference -> $dest"
    run "cd $REF && POPCORN_FD=3 POPCORN_GPUS=8 TMPDIR=$REF/.tmp TORCH_EXTENSIONS_DIR=$REF/.ext timeout 1500 python3 eval.py benchmark cases_bench.txt 3>$REF/benchmark.popcorn.txt >$REF/benchmark.stdout.txt 2>$REF/benchmark.stderr.txt"
    save "$REF" benchmark "$dest"
    check_pass "$dest/benchmark.popcorn.txt" "reference bench"
}

rank1_pass() { # rank1_pass <mode> <cases> <label> <dest>
    local mode=$1 cases=$2 label=$3 dest=$4
    kfd_wait_clean 30 10 || die "node not clean before rank1 $label"
    LOG "rank1 $mode ($label) -> $dest"
    run "rm -f $R1/ipc_handles_rank*.bin $R1/*.pkl"
    docker exec -w "$R1" \
        -e PATH="$COMPAT/bin:/usr/local/bin:/usr/bin:/bin:/opt/rocm/bin" \
        -e PYTHONPATH="$COMPAT:$IRISDST" \
        -e PYTHONUNBUFFERED=1 \
        -e POPCORN_FD=3 -e POPCORN_GPUS=8 \
        -e TRITON_CACHE_DIR="$R1/.triton" \
        -e TORCHINDUCTOR_CACHE_DIR="$R1/.inductor" \
        -e TORCH_EXTENSIONS_DIR="$R1/.ext" \
        -e TMPDIR="$R1/.tmp" \
        -e HSA_ENABLE_COREDUMP=0 \
        -e AMDGCN_USE_BUFFER_OPS=0 \
        "$NAME" bash -lc "timeout --signal=TERM 1700 setsid python3 -u eval.py $mode $cases 3>$R1/$label.popcorn.txt >$R1/$label.stdout.txt 2>$R1/$label.stderr.txt"
    save "$R1" "$label" "$dest"
    local shim
    shim=$(grep -c SHIM_WAS_CALLED "$R1/$label.stderr.txt" 2>/dev/null || true)
    [ "${shim:-0}" -eq 0 ] || die "rank1 sudo shim fired ($shim)"
    check_pass "$dest/$label.popcorn.txt" "rank1 $label"
}

# ---------------------------------------------------------------- lease + clocks
LOG "acquiring GPU lease"
bash "$T/gpu_lease.sh" acquire aug13_exp01 3600 || die "no lease"
trap 'bash "$T/gpu_lease.sh" release aug13_exp01' EXIT

LOG "pinning clocks"
bash "$T/set_clocks.sh" pin 1900 | tail -4

# ---------------------------------------------------------------- S0: stage + gate
LOG "===== S0 stage: ours ====="
run "cd $ON/harness && python3 -c \"
import importlib.util
spec = importlib.util.spec_from_file_location('dhk_rt', 'build/dhk_rt.so')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('ipc_probe:', m.ipc_probe(0))
\"" || die "ipc probe failed"
run "rm -rf $OURS && mkdir -p $OURS/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $OURS/ && cp $SRC/cases.txt $OURS/cases_test.txt && cp $ON/harness/hk_submission.py $OURS/submission.py"
run "cat > $OURS/cases_bench.txt <<'EOF'
$CASES_BENCH
EOF
cat $OURS/cases_bench.txt"
kfd_wait_clean 30 10 || die "node not clean before ours test"
run "cd $OURS && $(ours_env 0) timeout 900 python3 eval.py test cases_test.txt 3>$OURS/test.popcorn.txt >$OURS/test.stdout.txt 2>$OURS/test.stderr.txt"
save "$OURS" test "$RAW/s0/ours_test"
check_pass "$RAW/s0/ours_test/test.popcorn.txt" "ours test"
bench_ours 0 "$RAW/s0/O"

LOG "===== S0 stage: reference ====="
run "rm -rf $REF && mkdir -p $REF/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $SRC/submission.py $REF/ && cp $SRC/cases.txt $REF/cases_test.txt"
run "cp $OURS/cases_bench.txt $REF/cases_bench.txt"
kfd_wait_clean 30 10 || die "node not clean before reference test"
run "cd $REF && POPCORN_FD=3 POPCORN_GPUS=8 TMPDIR=$REF/.tmp TORCH_EXTENSIONS_DIR=$REF/.ext timeout 900 python3 eval.py test cases_test.txt 3>$REF/test.popcorn.txt >$REF/test.stdout.txt 2>$REF/test.stderr.txt"
save "$REF" test "$RAW/s0/reference_test"
check_pass "$RAW/s0/reference_test/test.popcorn.txt" "reference test"
bench_reference "$RAW/s0/R"

LOG "===== S0 stage: rank1 ====="
run "rm -rf $R1 && mkdir -p $R1/.tmp && cp $SRC/eval.py $SRC/task.py $SRC/utils.py $SRC/reference.py $R1/ && cp $SRC/cases.txt $R1/cases_test.txt"
run "python3 $ON/tools/patch_rank1.py $RANK1SUB $R1/submission.py" || die "patch_rank1 refused"
run "cp $OURS/cases_bench.txt $R1/cases_bench.txt"
run "rm -rf $R1/.triton"
rank1_pass benchmark cases_bench.txt warm "$RAW/s0/rank1_warm_THROWAWAY"
rank1_pass test cases_test.txt test "$RAW/s0/rank1_test"
rank1_pass benchmark cases_bench.txt bench "$RAW/s0/K"

# ---------------------------------------------------------------- sessions
session() { # session <name> <order...>
    local name=$1; shift
    LOG "===== session $name: $* ====="
    for arm in "$@"; do
        case "$arm" in
            O) bench_ours 0 "$RAW/$name/O" ;;
            D) bench_ours 1 "$RAW/$name/D" ;;
            R) bench_reference "$RAW/$name/R" ;;
            K) rank1_pass benchmark cases_bench.txt bench "$RAW/$name/K" ;;
            *) die "unknown arm $arm" ;;
        esac
    done
}

session s1 K R O D
session s2 D O R K
session s3 K O R
session s4 O K R

LOG "===== all sessions complete ====="
LOG "raw artifacts under $RAW"
