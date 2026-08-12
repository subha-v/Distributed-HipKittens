#!/usr/bin/env bash
# t06: STEP 1 -- re-gate the ratchet with the NaN poison on.
# Point 1 = the exact ratchet config (the re-gate itself).
# Point 2 = same + timestamps=1, a poison-on M6 stamp reference measured on the
#           pre-activation (donor-include) code path, for step 3 to compare to.
set -uo pipefail
rm -rf "$HOME/~"          # stray dir from a push with a literal tilde
mkdir -p "$HOME/overnight-scratch"
grep -n 'MPS SOAK..\] completed' "$HOME/tools/screen.sh" | head -3
LOG="$HOME/overnight-scratch/e32regate.driver.log"
cat > /tmp/e32regate.cfg <<'EOF'
C=16,g=353,mode=12,flush_rows=16
C=16,g=353,mode=12,flush_rows=16,timestamps=1
EOF
echo "== pre-flight =="
pgrep -af 'torchrun|mpirun' || echo "no torchrun/mpirun"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1" "$2}'
rm -f "$LOG"
setsid nohup env \
  SCREEN_ARMS=production,pf6gm_mega,mps_mega \
  SCREEN_WARMUP=1 SCREEN_TIMED=1 SCREEN_PROCS=1 \
  SCREEN_JOB_TIMEOUT=2400 SCREEN_RUN_TIMEOUT=2100 SCREEN_SYNC=1 \
  bash "$HOME/tools/screen.sh" e32regate /tmp/e32regate.cfg \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
sleep 20
echo "== first 40 lines =="
head -40 "$LOG"
exit 0
