#!/usr/bin/env bash
# exp_27 campaign pairing. The first candidate campaign read 6,477.0 us / 0.8400x
# against the 6,568.0 us / 0.8522x ratchet -- a 1.4% delta, which the experiment
# discipline says must be re-run or paired. So do BOTH, in this session, on the
# same node, alternating arms:
#   campaign 2 = CONTROL   (arm 0, d3d22ce4)  -> the ratchet measured HERE, today
#   campaign 3 = CANDIDATE (arm 1, f113d73f)  -> the replicate
# A same-session paired campaign A/B is the only valid denominator.
set -uo pipefail
DHK="$HOME/Distributed-HipKittens"
MPSSRC="$DHK/distributed-kernels/fused_moe/k0pf6gm_device_tile_mps.hip"

run_campaign () {
  local TAG="$1" SHA="$2" WANT="$3"
  for i in $(seq 1 120); do
    pgrep -f 'tools/screen.sh' >/dev/null 2>&1 || break
    [ "$i" = 1 ] && echo "  waiting for the running job ..."
    sleep 10
  done
  echo "################ $TAG  (arm $WANT, $SHA) ################"
  git -C "$DHK" fetch --all -q
  git -C "$DHK" reset -q --hard "$SHA"
  GOT="$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$MPSSRC" | awk '{print $3}')"
  echo "  node HEAD=$(git -C "$DHK" rev-parse --short HEAD)  ASCALE_TM=$GOT (want $WANT)"
  [ "$GOT" = "$WANT" ] || { echo "  REFUSING: wrong arm"; return 10; }
  /opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "  KFD: "$1" "$2}'
  echo "C=16,g=353,mode=12,flush_rows=16" > "/tmp/${TAG}.cfg"
  local LOG="$HOME/overnight-scratch/${TAG}.driver.log"
  rm -f "$LOG"
  setsid -w timeout 6300 env \
    SCREEN_ARMS=production,pf6gm_mega,mps_mega \
    SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
    SCREEN_JOB_TIMEOUT=6000 SCREEN_RUN_TIMEOUT=3000 SCREEN_SYNC=0 \
    bash "$HOME/tools/screen.sh" "$TAG" "/tmp/${TAG}.cfg" \
    > "$LOG" 2>&1 < /dev/null
  echo "  rc=$? ; tail:"
  grep -E '^SCREEN 1,' "$LOG" | cut -c1-190
  bash "$HOME/tools/e27_fp.sh" | grep -E 'TEXT sha|ARM PROBE|launched arm'
  echo
}

run_campaign e27camp2_ctl  d3d22ce4 0
run_campaign e27camp3_cand f113d73f 1
echo "################ ALL CAMPAIGNS DONE ################"
