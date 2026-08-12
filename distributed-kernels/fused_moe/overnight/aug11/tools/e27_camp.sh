#!/usr/bin/env bash
# exp_27 promotion campaign. The stamp ladder cleared (delta M6 = -78.8 us,
# t = -13.9, with the two control batches agreeing to +5.1 us), so this is
# step 4: five rotated processes, 500 warmup / 100 timed, all three arms, on the
# CANDIDATE arm pinned to f113d73f.
set -uo pipefail
for i in $(seq 1 90); do
  pgrep -f 'tools/screen.sh' >/dev/null 2>&1 || break
  [ "$i" = 1 ] && echo "waiting for the running batch to finish ..."
  sleep 10
done
echo "== batch 4 final rows =="
tail -4 "$HOME/overnight-scratch/e27b4_cand.driver.log" 2>/dev/null | cut -c1-150
echo
echo "################ FINGERPRINT OF THE ARM BATCH 4 LAUNCHED ################"
bash "$HOME/tools/e27_fp.sh"

echo
echo "################ CAMPAIGN: candidate arm 1, 5 rotations, 500w/100t ################"
TAG=e27camp_cand
DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard f113d73f
GOT="$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC" | awk '{print $3}')"
echo "node HEAD=$(git -C "$DHK" rev-parse --short HEAD)  ASCALE_TM=$GOT (want 1)"
[ "$GOT" = "1" ] || { echo "REFUSING: wrong arm"; exit 10; }

/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
echo "C=16,g=353,mode=12,flush_rows=16" > /tmp/${TAG}.cfg
LOG="$HOME/overnight-scratch/${TAG}.driver.log"
rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_JOB_TIMEOUT=6000 SCREEN_RUN_TIMEOUT=3000 SCREEN_SYNC=0 \
  bash "$HOME/tools/screen.sh" "$TAG" /tmp/${TAG}.cfg \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 25
head -18 "$LOG"
