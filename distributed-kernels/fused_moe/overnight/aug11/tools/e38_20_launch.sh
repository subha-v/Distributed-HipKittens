#!/usr/bin/env bash
# exp_38 GPU confirmation launcher.  usage: e38_20_launch.sh TAG CFGFILE PIN
#
# Refuses to start unless the pinned node checkout is BYTE-IDENTICAL to the two
# files the .text gate was run on. That is the whole provenance chain: gated
# sources -> gated .text == rev 26 -> these sources -> this campaign.
set -uo pipefail
TAG="${1:?usage: e38_20_launch.sh TAG CFGFILE PIN}"
CFGFILE="${2:?CFGFILE is mandatory}"
PIN="${3:?PIN is mandatory}"

DHK="$HOME/Distributed-HipKittens"
FM="$DHK/distributed-kernels/fused_moe"
GATED="$HOME/overnight-scratch/e38/tu/DEF"
LOG="$HOME/e38/${TAG}.driver.log"
mkdir -p "$HOME/e38"

echo "== exp_38 pre-flight =="; date -u
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print "KFD: "$1" "$2}'
if pgrep -f 'screen.sh' >/dev/null 2>&1; then
  echo "REFUSING: a screen batch is already running"; exit 9
fi

echo "== pin node checkout to $PIN =="
git -C "$DHK" fetch --all -q
git -C "$DHK" reset -q --hard "$PIN" || { echo "REFUSING: cannot reset to $PIN"; exit 11; }
echo "node HEAD  = $(git -C "$DHK" rev-parse HEAD)"
git -C "$DHK" status --porcelain | head -5

echo "== PROVENANCE: node checkout must equal the GATED sources =="
fail=0
for f in k0pf6gm_device_tile_mps.hip moe_mps_adapter.cuh; do
  a="$(sha256sum "$FM/$f" | cut -d' ' -f1)"
  b="$(sha256sum "$GATED/$f" | cut -d' ' -f1)"
  if [ "$a" = "$b" ]; then s="OK   "; else s="FAIL "; fail=1; fi
  printf '  %s %-34s checkout=%s gated=%s\n' "$s" "$f" "${a:0:16}" "${b:0:16}"
done
[ "$fail" = 0 ] || { echo "REFUSING: the checkout is not the binary we gated"; exit 12; }

echo "== knob state in the pinned tree =="
printf '  SRC_REV       = %s (want 30)\n' \
  "$(grep -oE '^#define K0P6_MPS_SRC_REV [0-9]+' "$FM/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')"
printf '  ASCALE_TM     = %s (want 1)\n' \
  "$(grep -oE '^#define K0P6_MPS_ASCALE_TM [0-9]+' "$FM/k0pf6gm_device_tile_mps.hip" | awk '{print $3}')"
grep -E '^#define (K0P6_MPS_ENABLE_MODE14|K0P6_MPS_E23_RING) ' "$FM/moe_mps_adapter.cuh" \
  | sed 's/^/  /'

echo "== configs (every line is a full 5-rotation campaign, STAMPS OFF) =="
grep -vE '^\s*(#|$)' "$CFGFILE" | nl

rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=500 SCREEN_TIMED=100 SCREEN_PROCS=5 \
  SCREEN_JOB_TIMEOUT=4500 SCREEN_RUN_TIMEOUT=1500 \
  SCREEN_SYNC=0 SCREEN_IDLE_WAIT=600 \
  bash "$HOME/tools/screen.sh" "$TAG" "$CFGFILE" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 25
head -40 "$LOG"
exit 0
