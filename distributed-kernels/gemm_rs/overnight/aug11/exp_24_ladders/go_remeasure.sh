#!/usr/bin/env bash
# exp_24 RE-MEASURE: instrument A only, at exp_26's shipped config.
#
# Why this exists as its own runner: the ladder measured earlier tonight ran on
# the pre-exp_26 module (79599cce...), so its 1.0971x graded / 1.1189x pipelined
# are the previous kernel's ratios. This re-runs instrument A ONLY -- no
# instrument B in this dispatch -- on fb3d670b..., which is exp_26's landed
# binary (verified three ways in go_prep2.sh + verify2.sh).
#
# Two changes from the previous run, and only two:
#   1. the module under test (the point of the exercise);
#   2. LAD_ROT=shuffle. The inherited cyclic order is defective -- it holds the
#      RELATIVE offset between every arm pair constant across reps, so a
#      neighbour effect cannot average out, and it had ours_null pinned
#      permanently one slot behind ours. See ladder_mp.py for the full note.
#
# That second change means the five shapes whose rgroup did NOT move are doing
# double duty: they are the control for the kernel change AND the control for
# the instrument change. If they come back flat, both are clean and shape 5's
# delta is attributable. If they move, the confound is real and the follow-up is
# LAD_ROT=cyclic on the same binary, which separates the two in one run.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
LEASE=$ON/tools/gpu_lease.sh
TAG=exp24r

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=========== module identity under test ==========="
SHA=$(sha256sum "$ON/harness/build/gemm_rs_mi300x.so" | cut -d' ' -f1)
log "gemm_rs_mi300x.so $SHA"
case "$SHA" in
  79599cce*) log "FATAL: pre-exp_26 binary; refusing to measure"; exit 1;;
esac
log "differs from pre-exp_26 79599cce... OK"

log "=========== acquire the lease ==========="
bash "$LEASE" acquire "$TAG" 14400 || { log "FATAL: lease not acquired"; exit 2; }
trap 'log "releasing lease"; bash "$LEASE" release "$TAG" || true' EXIT

log "=========== clocks ==========="
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -3

log "=========== instrument A, six shapes, five arms, shuffled order ==========="
# LAD_KEEP=0: the previous run's samples are archived under prev_79599cce/, so
# the new run starts from an empty sample set rather than pooling two binaries'
# measurements into one distribution.
# run_ladders.sh takes the phase as $1 (`PHASE=${1:-all}`), NOT from the
# environment. Passing it as an env var would have silently defaulted to "all"
# and run the hours-long instrument B this dispatch explicitly excludes.
export LAD_KEEP=0
export LAD_LEASED=1
export LAD_ROT=shuffle
export LAD_SHAPES=${LAD_SHAPES:-4,5,0,2,1,3}
export LAD_EXPECT_A=6
export LAD_TMO=${LAD_TMO:-2400}
setsid timeout --signal=TERM 10800 bash "$D/run_ladders.sh" A \
  > "$D/logs/remeasure_run.log" 2>&1
rc=$?
log "run_ladders.sh (phase A) rc=$rc"
tail -40 "$D/logs/remeasure_run.log"
exit $rc
