#!/usr/bin/env bash
# The decisive paired test: ours (PERSHAPE=2) vs ours_prev (PERSHAPE=0), same
# pool, same round, both protocols, both allocation orders, six shapes.
#
# Shape order puts 5 first -- it is the row under test and the one worth having
# if anything cuts the run short. The other five shapes are controls where the
# two rules agree, so they double as five extra null arms.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
RAW=$D/raw/ab_prev
LEASE=$ON/tools/gpu_lease.sh
TAG=exp24ab

ROUNDS=${AB_ROUNDS:-42}      # 7 complete blocks of 3! = 6 permutations; exp_26 used 40
GITERS=${AB_GITERS:-25}
PITERS=${AB_PITERS:-15}
BURST=${AB_BURST:-5}
SHAPES=${AB_SHAPES:-4,5,0,1,2,3}

log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$RAW" "$D/logs"

log "=========== build the three arms from the current source ==========="
bash "$D/build_ab_arms.sh" > "$D/logs/ab_build.log" 2>&1
rc=$?
grep -E '^(  OK|  FAIL|#####|  ASSERT|  FATAL|ab0|ab2)' "$D/logs/ab_build.log" | sed 's/^/  /'
if [ $rc -ne 0 ]; then
  log "FATAL: arm build failed (rc=$rc); last lines:"; tail -20 "$D/logs/ab_build.log"; exit 1
fi
log "arm build rc=0"

log "=========== lease ==========="
bash "$LEASE" acquire "$TAG" 14400 || { log "FATAL: lease not acquired"; exit 2; }
trap 'log "releasing lease"; bash "$LEASE" release "$TAG" || true' EXIT
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -2

log "=========== the campaign ==========="
IFS=',' read -ra LIST <<< "$SHAPES"
fail=0
for s in "${LIST[@]}"; do
  for order in fwd rev; do
    t0=$(date +%s)
    # Plain arithmetic: a $(...) inside $((...)) inside a docker exec argument is
    # the kind of nesting that has already cost this tree hours.
    if [ "$order" = "fwd" ]; then off=0; else off=1; fi
    port=$((13300 + s * 2 + off))
    log "---- shape index $s, allocation order $order, port $port ----"
    docker exec \
      -e PYTHONUNBUFFERED=1 \
      -e HK_DEBUG=0 \
      -e HK_BUILD_DIR="$ON/harness/build" \
      -e HSA_ENABLE_COREDUMP=0 \
      -e VS_FORCE_BIAS=1 \
      -e AB_ORDER="$order" \
      -e AB_OUT="$RAW/ab_s${s}_${order}" \
      -e AB_WARM_MS="${AB_WARM_MS:-400}" \
      -w "$D" dhk-gemmrs bash -lc \
      "timeout --signal=TERM ${AB_TMO:-2400} setsid python3 -u $D/ab_prev_mp.py \
       $s $ROUNDS $GITERS $PITERS $BURST $port" \
      2>&1 | tee "$D/logs/ab_s${s}_${order}.log" | grep -E 'rank0|exit codes|Error|Traceback|FATAL'
    log "shape $s/$order wall=$(( $(date +%s) - t0 ))s"
    grep -q 'exit codes: \[0, 0, 0, 0, 0, 0, 0, 0\]' "$D/logs/ab_s${s}_${order}.log" \
      || { log "WARN: shape $s/$order did not return all-zero exit codes"; fail=1; }
  done
done

log "=========== score against the pre-registered outcomes ==========="
python3 "$D/ab_prev_report.py" 2>&1 | tee "$D/ab_prev_report.txt"
log "campaign fail flag=$fail"
exit 0
