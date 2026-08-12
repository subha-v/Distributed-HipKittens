#!/usr/bin/env bash
# t16 (pushed to ~/tools/ladder.sh): run N repeats of the ratchet config with
# timestamps on, as one screen batch. Usage: ladder.sh TAG NREPS EXPECT_MASK
#
# One batch == one build == one mask, because the phase-1 hint mask is a
# compile-time #define. The sync is done HERE, before the batch, and screen.sh
# is then run with SCREEN_SYNC=0 so nothing can move mid-batch.
#
# EXPECT_MASK is mandatory and is the whole point of this rewrite: a batch that
# was tagged e26_m5 once ran mask 1, because another agent's push landed first
# and my commit was rejected while the launch went ahead anyway. The node
# checkout is the arm, so the arm is now verified against the tag BEFORE any GPU
# work, and the batch refuses to start on a mismatch.
set -uo pipefail
TAG="${1:?usage: ladder.sh TAG NREPS EXPECT_MASK}"
N="${2:-5}"
EXPECT="${3:?EXPECT_MASK is mandatory}"
CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1"
DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
BRANCH=codex/distributed-hipkittens-scaffold
LOG="$HOME/overnight-scratch/${TAG}.driver.log"

echo "== pre-flight =="
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
if pgrep -f 'tools/screen.sh' >/dev/null 2>&1; then
  echo "REFUSING: a screen batch is already running"; exit 9
fi

echo "== sync node checkout to origin/$BRANCH (once, before the batch) =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard "origin/$BRANCH"
HEADSHA="$(git -C "$DHK" rev-parse --short HEAD)"
GOT="$(grep -oE '^#define N2GM_P1_SCHED_GSCALE [0-9]+' "$MPSSRC" | awk '{print $3}')"
SRCREV="$(grep -oE 'K0P6_MPS_SRC_REV [0-9]+' "$MPSSRC" | tail -1 | awk '{print $2}')"
INC="$(grep -oE '^#include "n2_phase1_gm(_mps)?\.cpp"' "$MPSSRC" | tail -1)"
echo "node HEAD=$HEADSHA  SRC_REV=$SRCREV  mask=$GOT  include=$INC"
if [ "$GOT" != "$EXPECT" ]; then
  echo "REFUSING: node checkout carries mask '$GOT' but tag '$TAG' expects '$EXPECT'."
  echo "  (did a push get rejected? re-push and retry)"
  exit 10
fi

: > /tmp/${TAG}.cfg
for i in $(seq 1 "$N"); do echo "$CFG" >> /tmp/${TAG}.cfg; done
echo "$TAG" > /tmp/screen_tag
echo "== $N repeats of: $CFG  (mask $GOT) =="

rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1 \
  SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100 SCREEN_SYNC=0 \
  bash "$HOME/tools/screen.sh" "$TAG" /tmp/${TAG}.cfg \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 20
head -20 "$LOG"
exit 0
