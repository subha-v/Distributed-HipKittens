#!/usr/bin/env bash
# exp_36 step 10: launch one screen batch at a given T, detached.
#   env: E36_T (tokens per rank), E36_TAG, E36_CFGS (newline list, base64),
#        E36_PROCS/E36_WARMUP/E36_TIMED (default screen 1/1/1)
set -u
E36="$HOME/e36"
T="${E36_T:-4096}"
TAG="${E36_TAG:-e36T${T}}"
PROCS="${E36_PROCS:-1}"
WARMUP="${E36_WARMUP:-1}"
TIMED="${E36_TIMED:-1}"
JOBTO="${E36_JOB_TIMEOUT:-5400}"
CFGFILE="$E36/${TAG}.cfgs"
LOG="$E36/${TAG}.batchlog"

mkdir -p "$E36/scratch"
if [ -n "${E36_CFGS_B64:-}" ]; then
  printf '%s' "$E36_CFGS_B64" | base64 -d > "$CFGFILE"
fi
if [ ! -s "$CFGFILE" ]; then echo "no cfgs at $CFGFILE" >&2; exit 2; fi
if [ -e "$LOG" ]; then echo "log exists: $LOG (pick a new tag)" >&2; exit 3; fi

echo "=== cfgs ==="; cat "$CFGFILE"; echo "============"

# CAPMODE=pinned (default): every buffer capacity stays at the T=4096 value, so
# the ONLY variable is how many tokens actually flow. CAPMODE=scaled: capacities
# follow T by the harness's own formulas (a second, deployment-realistic arm).
CAP="${E36_CAPMODE:-pinned}"
if [ "$CAP" = "pinned" ]; then
  CAP_TLOC=40960; CAP_PAD=263136; CAP_MAXTOK=4096; CAP_MAXTOK_PROD=4096
else
  CAP_TLOC=""; CAP_PAD=""; CAP_MAXTOK=""; CAP_MAXTOK_PROD=""
fi
echo "capmode=$CAP T_LOC_MAX=${CAP_TLOC:-auto} PADMAX=${CAP_PAD:-auto} MAXTOK=${CAP_MAXTOK:-auto}"

cd "$E36"
SCREEN_CAMPAIGN="$E36/rc_T.sh" \
SCREEN_SCRATCH="$E36/scratch" \
SCREEN_SYNC=0 \
SCREEN_T="$T" \
SCREEN_T_LOC_MAX="$CAP_TLOC" SCREEN_PADMAX="$CAP_PAD" \
SCREEN_MAXTOK="$CAP_MAXTOK" SCREEN_MAXTOK_PROD="$CAP_MAXTOK_PROD" \
SCREEN_PROCS="$PROCS" SCREEN_WARMUP="$WARMUP" SCREEN_TIMED="$TIMED" \
SCREEN_JOB_TIMEOUT="$JOBTO" SCREEN_RUN_TIMEOUT="${E36_RUN_TIMEOUT:-2400}" \
setsid nohup timeout "$((JOBTO * 4))" bash "$E36/screenT.sh" "$TAG" "$CFGFILE" \
  > "$LOG" 2>&1 < /dev/null &
disown
sleep 3
echo "launched pid=$! tag=$TAG T=$T log=$LOG"
head -20 "$LOG" 2>/dev/null
exit 0
