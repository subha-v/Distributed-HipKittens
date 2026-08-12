#!/usr/bin/env bash
# Smoke test for the paired A/B: does loading three differently-compiled kernel
# modules into ONE process actually work? That is the novel mechanism here --
# submission.py binds HK_KERNEL_MODULE at import, so three imports under three
# names must yield three independent arms, each doing its own IPC exchange.
# Cheap on purpose: 6 rounds, 3 iters. Validates the mechanism, not the numbers.
set -uo pipefail
ON=/home/subvadla/dhk/distributed-kernels/gemm_rs/overnight
D=$ON/aug11/exp_24_ladders
RAW=$D/raw/ab_smoke
LEASE=$ON/tools/gpu_lease.sh
TAG=exp24abs
mkdir -p "$RAW" "$D/logs"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=========== build arms ==========="
bash "$D/build_ab_arms.sh" > "$D/logs/ab_build.log" 2>&1
rc=$?
grep -E '^(  OK|  FAIL|  FATAL|  ab0|  ab2|##########)' "$D/logs/ab_build.log" | sed 's/^/  /'
[ $rc -ne 0 ] && { log "FATAL build rc=$rc"; tail -25 "$D/logs/ab_build.log"; exit 1; }

log "=========== lease ==========="
bash "$LEASE" acquire "$TAG" 3600 || { log "FATAL lease"; exit 2; }
trap 'log "releasing lease"; bash "$LEASE" release "$TAG" || true' EXIT
bash "$ON/tools/set_clocks.sh" pin 1900 2>&1 | tail -1

log "=========== smoke: shape 5, 6 rounds ==========="
docker exec -e PYTHONUNBUFFERED=1 -e HK_DEBUG=0 \
  -e HK_BUILD_DIR="$ON/harness/build" -e HSA_ENABLE_COREDUMP=0 \
  -e VS_FORCE_BIAS=1 -e AB_ORDER=fwd -e AB_OUT="$RAW/ab_s4_fwd" \
  -w "$D" dhk-gemmrs bash -lc \
  "timeout --signal=TERM 900 setsid python3 -u $D/ab_prev_mp.py 4 6 3 3 2 13399" \
  2>&1 | tee "$D/logs/ab_smoke.log" \
  | grep -E 'rank0|exit codes|Error|Traceback|FATAL|refusing|NOT bit'
log "smoke rc=$?"

echo
log "=========== what the smoke produced ==========="
ls -la "$RAW" | tail -10
for f in "$RAW"/ab_s4_fwd.rank0.json; do
  [ -f "$f" ] && python3 "$D/ab_smoke_check.py" "$f"
done
