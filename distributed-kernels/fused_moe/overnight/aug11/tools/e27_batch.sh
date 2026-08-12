#!/usr/bin/env bash
# exp_27 batch launcher.  usage: e27_batch.sh TAG SHA EXPECT_TM NREPS
#
# The exp_27 arm is a LITERAL in the hashed .hip, so control and candidate cannot
# be interleaved inside one batch -- one batch is one build is one arm. This
# launcher therefore pins the node checkout to an EXPLICIT COMMIT rather than to
# origin/HEAD, so the four batches can alternate arms without a commit per batch
# and without caring what other agents push to the branch meanwhile.
#
# It refuses to start unless the literal in the pinned checkout equals EXPECT_TM.
# A batch already ran the wrong arm tonight because a push was rejected while the
# launch went ahead; the arm is verified against the tag BEFORE any GPU work.
set -uo pipefail
TAG="${1:?usage: e27_batch.sh TAG SHA EXPECT_TM NREPS}"
SHA="${2:?SHA is mandatory -- pin the arm, do not trust origin/HEAD}"
EXPECT="${3:?EXPECT_TM is mandatory (0 = control, 1 = candidate)}"
N="${4:-5}"

CFG="C=16,g=353,mode=12,flush_rows=16,timestamps=1"
DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
P1SRC="$DHK/distributed-kernels/fused_moe/n2_phase1_gm_mps.cpp"
LOG="$HOME/overnight-scratch/${TAG}.driver.log"
mkdir -p "$HOME/overnight-scratch"

echo "== pre-flight =="
date -u
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
if pgrep -f 'screen.sh' >/dev/null 2>&1; then
  echo "REFUSING: a screen batch is already running"; exit 9
fi

echo "== pin node checkout to $SHA =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard "$SHA" || { echo "REFUSING: cannot reset to $SHA"; exit 11; }
HEADSHA="$(git -C "$DHK" rev-parse --short HEAD)"
GOT="$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC" | awk '{print $3}')"
SRCREV="$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$MPSSRC" | awk '{print $3}')"
INC="$(grep -oE '^#include "n2_phase1_gm(_mps)?\.cpp"' "$MPSSRC" | tail -1)"
KNOB="$(grep -oE '^#define N2GM_P1_ASCALE_TOKEN_MAJOR [0-9]+' "$P1SRC" | awk '{print $3}')"
# -A14, not -A8: the exp_27 comment block sits between the call and the #if, so a
# window of 8 stopped short of the slot lines and printed an empty probe.
SLOT="$(grep -A14 'n2p6gm_phase1_body(' "$MPSSRC" | grep -oE 'K0P6_D_SC_(DST|STAGE)' | tr '\n' ' ')"
echo "node HEAD=$HEADSHA  SRC_REV=$SRCREV  ASCALE_TM=$GOT (want $EXPECT)"
echo "  phase-1 include : $INC"
echo "  p1 knob default : $KNOB (must be 0 -- the .hip literal is what governs)"
echo "  M6 slot arms    : $SLOT"
echo "  .hip sha256     : $(sha256sum "$MPSSRC" | cut -c1-16)"
echo "  p1   sha256     : $(sha256sum "$P1SRC" | cut -c1-16)"
if [ "$GOT" != "$EXPECT" ]; then
  echo "REFUSING: checkout carries ASCALE_TM='$GOT' but tag '$TAG' expects '$EXPECT'."
  exit 10
fi
if [ "$KNOB" != "0" ]; then
  echo "REFUSING: the vendored p1 default is not 0, so the .hip literal is not the only source of truth."
  exit 12
fi

: > "/tmp/${TAG}.cfg"
for i in $(seq 1 "$N"); do echo "$CFG" >> "/tmp/${TAG}.cfg"; done
echo "== $N repeats of: $CFG   (arm ASCALE_TM=$GOT) =="

rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1 \
  SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100 SCREEN_SYNC=0 \
  bash "$HOME/tools/screen.sh" "$TAG" "/tmp/${TAG}.cfg" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 25
head -24 "$LOG"
exit 0
