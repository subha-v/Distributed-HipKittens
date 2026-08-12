#!/usr/bin/env bash
# exp_27 close-out: leave the node checkout at origin HEAD and prove that HEAD
# carries the WINNING arm, so the next experiment inherits the new ratchet.
set -uo pipefail
DHK="$HOME/Distributed-HipKittens"
FM="$DHK/distributed-kernels/fused_moe"
MPSSRC="$FM/k0pf6gm_device_tile_mps.hip"
BRANCH=codex/distributed-hipkittens-scaffold

echo "== node idle? =="
pgrep -af 'torchrun|mpirun|tools/screen.sh' || echo "no torchrun/mpirun/screen.sh"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'

echo
echo "== sync node checkout to origin/$BRANCH =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard "origin/$BRANCH"
git -C "$DHK" log --oneline -1
echo "node HEAD   : $(git -C "$DHK" rev-parse HEAD)"
echo "origin HEAD : $(git -C "$DHK" rev-parse origin/$BRANCH)"

echo
echo "== does HEAD carry the WINNING arm? =="
grep -nE '^#define K0P6_MPS_SRC_REV [0-9]+|^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC"
echo "--- M6's phase-1 A_scale argument, both arms ---"
grep -n -A14 'n2p6gm_phase1_body(' "$MPSSRC" | grep -E 'K0P6_D_SC_(DST|STAGE)|#if|#else|#endif'
echo "--- M5: is scale_transpose_row still reachable? ---"
grep -n 'scale_transpose_row\|zero_part_scale_transpose\|K0P6_MPS_ASCALE_TM' "$MPSSRC" | sed -n '1,12p'

echo
echo "== the shipping arm, restated =="
TM="$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC" | awk '{print $3}')"
if [ "$TM" = "1" ]; then
  echo "  OK: origin HEAD ships K0P6_MPS_ASCALE_TM=1 -- the 6,482.7 us / 0.8408x arm."
else
  echo "  ** WARNING: origin HEAD ships ASCALE_TM=$TM, NOT the winning arm **"
fi
echo
echo "== all three campaigns, final =="
bash "$HOME/tools/e27_camps.sh"
