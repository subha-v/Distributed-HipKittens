#!/usr/bin/env bash
# exp_24 batch e24b: e24a's runs 4-6 drifted 6.6% on a repeated control, so the
# depth axis needs a re-read with (a) three interleaved controls to bracket any
# time-correlated drift and (b) timestamps=1 on EVERY point, because the throttle
# acts on M7 and the M7 stamp is far more sensitive than the end-to-end ratio.
set -uo pipefail
SC=$HOME/.overnight-scripts
CFG=$HOME/overnight-scratch/e24b.cfgs
cat > "$CFG" <<'EOF'
C=16,g=33,mode=12,flush_rows=16,timestamps=1
C=16,g=289,mode=12,flush_rows=16,timestamps=1
C=16,g=545,mode=12,flush_rows=16,timestamps=1
C=16,g=801,mode=12,flush_rows=16,timestamps=1
C=16,g=33,mode=12,flush_rows=16,timestamps=1
C=16,g=97,mode=12,flush_rows=16,timestamps=1
C=16,g=353,mode=12,flush_rows=16,timestamps=1
C=16,g=97,mode=12,flush_rows=16,timestamps=1
C=16,g=33,mode=12,flush_rows=16,timestamps=1
EOF
echo "=== foreign-activity forensics for the e24a window (04:38-04:43Z)"
echo "-- docker containers started recently"
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.CreatedAt}}' 2>/dev/null | head -12
echo "-- other tenants' output roots touched in the last hour"
find $HOME -maxdepth 1 -name 'k0-mok-*' -newermt '-70 minutes' 2>/dev/null | head
echo "-- any non-gpuagent KFD process right now"
/opt/rocm/bin/rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ {print $1, $2}'
echo "-- load average / uptime"
uptime
echo
echo "=== launching e24b detached (9 points, all timestamps=1)"
cd "$HOME/amd-master/auto-gpu-kernel/k0_fused_moe"
SCREEN_SYNC=0 setsid nohup timeout 5400 \
  bash "$SC/screen.sh" e24b "$CFG" \
  > "$HOME/overnight-scratch/e24b_batch.out" 2>&1 < /dev/null &
echo "batch pid=$!"
sleep 15
head -12 "$HOME/overnight-scratch/e24b_batch.out"
exit 0
